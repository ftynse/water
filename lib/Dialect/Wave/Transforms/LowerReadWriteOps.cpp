// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

/// Materialize affine.apply for expressions inside a `map` with `symbols`.
/// Each symbol is either a GPU id (thread/block) or a constant from `hyper`.
static FailureOr<SmallVector<Value>>
materializeAffine(Location loc, ArrayRef<wave::WaveSymbolAttr> symbols,
                  AffineMap map, PatternRewriter &rewriter,
                  wave::WaveHyperparameterAttr hyper) {
  // NOTE: This helper assumes 0 dims in `map`. If you add dims, prepend
  // the dim operands before the symbol operands below.
  assert(map.getNumDims() == 0 && "expected 0 dims");

  auto threadId = [&](gpu::Dimension d) -> Value {
    return rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(), d);
  };
  auto blockId = [&](gpu::Dimension d) -> Value {
    return rewriter.create<gpu::BlockIdOp>(loc, rewriter.getIndexType(), d);
  };

  SmallVector<Value> baseSymVals;
  baseSymVals.reserve(map.getNumSymbols());
  int64_t numSym = map.getNumSymbols();
  for (int64_t i = 0; i < numSym; ++i) {
    StringRef name = symbols[i].getName();

    Value v;
    if (name == "T0")
      v = threadId(gpu::Dimension::x);
    else if (name == "T1")
      v = threadId(gpu::Dimension::y);
    else if (name == "T2")
      v = threadId(gpu::Dimension::z);
    else if (name == "WG0")
      v = blockId(gpu::Dimension::x);
    else if (name == "WG1")
      v = blockId(gpu::Dimension::y);
    else if (name == "WG2")
      v = blockId(gpu::Dimension::z);
    else if (std::optional<int64_t> value = hyper.getSymbolValue(name)) {
      v = rewriter.create<arith::ConstantIndexOp>(loc, *value);
    } else {
      emitError(loc) << "unknown hyperparameter symbol '" << name << "'";
      return failure();
    }
    baseSymVals.push_back(v);
  }
  // In case map contains multiple results, create one apply per result
  SmallVector<Value> results;
  results.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    AffineMap submap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), expr);
    llvm::SmallVector<Value> symVals = baseSymVals;
    affine::canonicalizeMapAndOperands(&submap, &symVals);

    Value apply = rewriter.create<affine::AffineApplyOp>(loc, submap, symVals);
    results.push_back(apply);
  }

  return results;
}

static SmallVector<Value>
buildStartIndices(Location loc, DictionaryAttr indexDict,
                  ArrayRef<wave::WaveSymbolAttr> orderedSyms,
                  PatternRewriter &rewriter,
                  wave::WaveHyperparameterAttr hyper) {
  SmallVector<Value> indices;
  indices.reserve(orderedSyms.size());
  // For each dimension in the memref (in order), find its corresponding start
  // index by symbol
  for (auto symAttr : orderedSyms) {
    StringRef name = symAttr.getName();
    Attribute a = indexDict.get(name);
    assert(a && "index dict missing entry for dimension symbol");
    auto mapAttr = cast<wave::WaveIndexMappingAttr>(a);

    FailureOr<SmallVector<Value>> startFo = materializeAffine(
        loc, mapAttr.getSymbolNames(), mapAttr.getStart(), rewriter, hyper);
    SmallVector<Value> start = std::move(*startFo);
    indices.push_back(start[0]); // start map has one result
  }
  return indices;
}

// Pick the vectorized (fastest) dimension based on the per-dimension SIZE
// from the index attribute. (largest size wins; tie → last dim)
static int64_t findFastestDimBySize(DictionaryAttr indexDict,
                                    wave::WaveHyperparameterAttr hyper) {

  SmallVector<NamedAttribute> entries(indexDict.begin(), indexDict.end());
  int64_t bestIdx = -1;
  std::optional<int64_t> bestSize; // largest constant size seen so far

  for (int64_t i = 0, e = (int64_t)entries.size(); i < e; ++i) {
    auto mapAttr =
        llvm::cast<wave::WaveIndexMappingAttr>(entries[i].getValue());
    std::optional<llvm::SmallVector<int64_t>> vals =
        wave::resolveSymbolNames(mapAttr.getSymbolNames(), hyper);
    std::optional<llvm::SmallVector<int64_t>> folded =
        wave::evaluateMapWithSymbols(mapAttr.getStep(), *vals);
    int64_t size = (*folded)[0];

    if (!bestSize || size > *bestSize || (size == *bestSize && i > bestIdx)) {
      bestSize = size;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/// Build a per-thread mask
///  mask = AND_d ( idx_d(lane) < bound_d )
static Value buildMask(Location loc, wave::DistributedShapeAttr boundsAttr,
                       PatternRewriter &rewriter, DictionaryAttr indexDict,
                       wave::WaveHyperparameterAttr hyper,
                       ArrayRef<Value> startIdx, int64_t elementsPerThread) {

  if (!boundsAttr)
    return Value();

  const int64_t rank = static_cast<int64_t>(startIdx.size());

  int64_t fastestDim = findFastestDimBySize(indexDict, hyper);

  IndexType idxType = rewriter.getIndexType();
  VectorType vecIdxType = VectorType::get({elementsPerThread}, idxType);
  IntegerType i1Type = IntegerType::get(rewriter.getContext(), 1);
  VectorType maskType = VectorType::get({elementsPerThread}, i1Type);

  // iota [0..L-1] : vector<index>
  Value iota = rewriter.create<mlir::vector::StepOp>(loc, vecIdxType);

  // Lane indices for fastest dim: start + iota
  Value startFastVec = rewriter.create<vector::BroadcastOp>(
      loc, vecIdxType, startIdx[fastestDim]);
  Value laneIdxFast = rewriter.create<arith::AddIOp>(loc, startFastVec, iota);

  // Materialize bounds
  FailureOr<SmallVector<Value>> boundValsFo = materializeAffine(
      loc, boundsAttr.getSymbolNames(), boundsAttr.getShape(), rewriter, hyper);
  SmallVector<Value> boundVals = std::move(*boundValsFo);

  // finalMask is the AND of per-dimension bound checks
  Value finalMask;
  for (int64_t d = 0; d < rank; ++d) {
    Value bound = boundVals[d];
    Value clause;
    if (d == fastestDim) {
      // lane-wise compare: (start + iota) < bound
      Value boundVec =
          rewriter.create<vector::BroadcastOp>(loc, vecIdxType, bound);
      clause = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                              laneIdxFast, boundVec);
    } else {
      // scalar compare then broadcast: startIdx[d] < bound
      Value scalarCmp = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, startIdx[d], bound);
      clause = rewriter.create<vector::BroadcastOp>(loc, maskType, scalarCmp);
    }

    finalMask =
        finalMask
            ? rewriter.create<arith::AndIOp>(loc, finalMask, clause).getResult()
            : clause;
  }

  return finalMask;
}

static Value buildVectorRead(Location loc, PatternRewriter &rewriter, Value mem,
                             ArrayRef<Value> indices, VectorType vecType,
                             Value mask) {

  if (!mask)
    return rewriter.create<vector::LoadOp>(loc, vecType, mem, indices);

  // Create a passthrough vector with elements set to zero corresponding to
  // the element type in memory.
  Type eltType = vecType.getElementType();
  Attribute zeroElement;
  if (auto flt = dyn_cast<FloatType>(eltType))
    zeroElement = rewriter.getFloatAttr(flt, 0.0);
  else if (auto it = dyn_cast<IntegerType>(eltType))
    zeroElement = rewriter.getIntegerAttr(it, 0);
  else
    assert(false && "unsupported element type");
  DenseElementsAttr zeroSplat = SplatElementsAttr::get(vecType, zeroElement);

  Value passthrough =
      rewriter.create<arith::ConstantOp>(loc, vecType, zeroSplat);
  return rewriter.create<vector::MaskedLoadOp>(loc, vecType, mem, indices, mask,
                                               passthrough);
}

static void buildVectorWrite(Location loc, PatternRewriter &rewriter, Value mem,
                             ArrayRef<Value> indices, Value vecValue,
                             Value mask) {
  if (mask) {
    rewriter.create<vector::MaskedStoreOp>(loc, mem, indices, mask, vecValue);
  } else {
    rewriter.create<vector::StoreOp>(loc, vecValue, mem, indices);
  }
}

wave::WaveHyperparameterAttr
getHyperparametersFromConverter(const mlir::TypeConverter *base) {
  auto &tc = static_cast<const wave::WaveTypeConverter &>(*base);
  return tc.getHyperparameters();
}

class ReadOpLoweringPattern : public OpConversionPattern<wave::ReadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::ReadOp op, wave::ReadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(op,
                                         "WaveTensorType conversion failed");

    auto vectorType = cast<VectorType>(convertedType);
    if (!vectorType)
      return rewriter.notifyMatchFailure(
          op, "expected vector type after conversion");

    int64_t elementsPerThread = vectorType.getNumElements();

    Value base = adaptor.getMemory();
    auto memrefTy = cast<MemRefType>(base.getType());
    if (!memrefTy)
      return rewriter.notifyMatchFailure(
          op, "expected memref base after conversion");

    auto memoryType = cast<wave::WaveTensorType>(op.getMemory().getType());
    ArrayRef<wave::WaveSymbolAttr> orderedSyms = memoryType.getShape();

    wave::DistributedShapeAttr boundsAttr = op.getBoundsAttr();
    DictionaryAttr index = op.getIndexAttr();

    wave::WaveHyperparameterAttr hyper =
        getHyperparametersFromConverter(getTypeConverter());

    // Build per-dimension start indices
    SmallVector<Value> indices =
        buildStartIndices(loc, index, orderedSyms, rewriter, hyper);

    Value mask = buildMask(loc, boundsAttr, rewriter, index, hyper, indices,
                           elementsPerThread);

    Value readOp =
        buildVectorRead(loc, rewriter, base, indices, vectorType, mask);

    rewriter.replaceOp(op, readOp);

    return success();
  }
};

class WriteOpLoweringPattern : public OpConversionPattern<wave::WriteOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::WriteOp op, wave::WriteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value base = adaptor.getMemory();
    auto memrefTy = cast<MemRefType>(base.getType());
    if (!memrefTy)
      return rewriter.notifyMatchFailure(op, "expected converted memref");

    Value vec = adaptor.getValueToStore();
    auto vecType = cast<VectorType>(vec.getType());
    if (!vecType)
      return rewriter.notifyMatchFailure(op, "expected vector value to store");

    auto memoryType = cast<wave::WaveTensorType>(op.getMemory().getType());
    ArrayRef<wave::WaveSymbolAttr> orderedSyms = memoryType.getShape();

    int64_t elementsPerThread = vecType.getNumElements();
    wave::DistributedShapeAttr boundsAttr = op.getBoundsAttr();
    DictionaryAttr index = op.getIndexAttr();

    wave::WaveHyperparameterAttr hyper =
        getHyperparametersFromConverter(getTypeConverter());

    SmallVector<Value> indices =
        buildStartIndices(loc, index, orderedSyms, rewriter, hyper);

    // Build per-lane mask (or none)
    Value mask = buildMask(loc, boundsAttr, rewriter, index, hyper, indices,
                           elementsPerThread);

    buildVectorWrite(loc, rewriter, base, indices, vec, mask);

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void wave::populateWaveReadWriteLoweringPatterns(
    WaveTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ReadOpLoweringPattern, WriteOpLoweringPattern>(
      typeConverter, patterns.getContext());
}
