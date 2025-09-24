#// Copyright 2025 The Water Authors
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
static llvm::SmallVector<mlir::Value>
materializeAffine(Location loc, ArrayRef<wave::WaveSymbolAttr> symbols,
                  AffineMap map, PatternRewriter &rewriter,
                  wave::WaveHyperparameterAttr hyper) {
  // NOTE: This helper assumes 0 dims in `map`. If you add dims, prepend
  // the dim operands before the symbol operands below.

  auto threadId = [&](gpu::Dimension d) -> Value {
    return rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(), d);
  };
  auto blockId = [&](gpu::Dimension d) -> Value {
    return rewriter.create<gpu::BlockIdOp>(loc, rewriter.getIndexType(), d);
  };

  SmallVector<Value> baseSymVals;
  baseSymVals.reserve(map.getNumSymbols());

  for (unsigned i = 0; i < map.getNumSymbols(); ++i) {
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
    else if (auto value = hyper.getSymbolValue(name)) {
      v = rewriter.create<arith::ConstantIndexOp>(loc, *value);
    } else {
      // emit error: Unknown symbol
    }
    baseSymVals.push_back(v);
  }
  // In case map contains multiple results, create one apply per result
  llvm::SmallVector<Value> results;
  results.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    AffineMap submap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), expr);
    llvm::SmallVector<Value> symVals = baseSymVals;
    affine::canonicalizeMapAndOperands(&submap, &symVals);

    Value apply = rewriter.create<affine::AffineApplyOp>(loc, submap, symVals)
                      .getResult();
    results.push_back(apply);
  }

  return results;
}

static SmallVector<Value>
buildStartIndices(Location loc, DictionaryAttr index, PatternRewriter &rewriter,
                  wave::WaveHyperparameterAttr hyper) {
  SmallVector<Value> indices;

  // TODO : The loop below currently builds `indices` using the iteration order
  // of the dictionary which is not guaranteed to match the memref's logical
  // dimension order
  for (NamedAttribute dim : index) {
    auto mapAttr = dyn_cast<wave::WaveIndexMappingAttr>(dim.getValue());
    SmallVector<Value> start = materializeAffine(
        loc, mapAttr.getSymbolNames(), mapAttr.getStart(), rewriter, hyper);
    indices.push_back(start[0]); // start map has one result
  }
  return indices;
}

// Pick the vectorized (fastest) dimension based on the per-dimension SIZE
// from the index attribute. (largest size wins; tie → last dim)
static int findFastestDimBySize(DictionaryAttr indexDict,
                                wave::WaveHyperparameterAttr hyper) {

  SmallVector<NamedAttribute> entries(indexDict.begin(), indexDict.end());
  int bestIdx = -1;
  std::optional<int64_t> bestSize; // largest constant size seen so far

  for (int i = 0, e = (int)entries.size(); i < e; ++i) {
    auto mapAttr =
        llvm::dyn_cast<wave::WaveIndexMappingAttr>(entries[i].getValue());
    auto vals = wave::resolveSymbolNames(mapAttr.getSymbolNames(), hyper);
    auto folded = wave::evaluateMapWithSymbols(mapAttr.getStep(), *vals);
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
static std::optional<mlir::Value>
buildMask(mlir::Location loc, wave::DistributedShapeAttr boundsAttr,
          mlir::PatternRewriter &rewriter, DictionaryAttr indexDict,
          wave::WaveHyperparameterAttr hyper,
          mlir::ArrayRef<mlir::Value> startIdx,
          int64_t elementsPerThread = 32) {

  if (!boundsAttr)
    return std::nullopt;

  const int64_t rank = static_cast<int64_t>(startIdx.size());

  int fastestDim = findFastestDimBySize(indexDict, hyper);

  auto idxType = rewriter.getIndexType();
  auto vecIdxType = VectorType::get({elementsPerThread}, idxType);
  auto i1Type = IntegerType::get(rewriter.getContext(), 1);
  auto maskType = VectorType::get({elementsPerThread}, i1Type);

  // iota [0..L-1] : vector<index>
  SmallVector<Attribute> lanes;
  lanes.reserve(elementsPerThread);
  for (int64_t i = 0; i < elementsPerThread; ++i)
    lanes.push_back(IntegerAttr::get(idxType, i));
  auto denseIota = DenseElementsAttr::get(vecIdxType, lanes);
  Value iota = rewriter.create<arith::ConstantOp>(loc, vecIdxType, denseIota);

  // Lane indices for fastest dim: start + iota
  Value startFastVec = rewriter.create<vector::BroadcastOp>(
      loc, vecIdxType, startIdx[fastestDim]);
  Value laneIdxFast = rewriter.create<arith::AddIOp>(loc, startFastVec, iota);

  // Materialize bounds
  SmallVector<Value> boundVals = materializeAffine(
      loc, boundsAttr.getSymbolNames(), boundsAttr.getShape(), rewriter, hyper);

  // finalMask is the AND of per-dimension bound checks
  Value finalMask;
  for (int d = 0; d < rank; ++d) {
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

static mlir::Value
buildVectorRead(mlir::Location loc, mlir::PatternRewriter &rewriter,
                mlir::Value mem, llvm::ArrayRef<mlir::Value> indices,
                mlir::VectorType vecType, std::optional<mlir::Value> maskOpt) {
  Value mask = maskOpt ? *maskOpt : mlir::Value{};

  if (mask) {
    // Create a passthrough vector with elements set to zero corresponding to
    // the element type in memory.
    auto eltType = vecType.getElementType();
    Attribute zeroElement;
    if (auto flt = dyn_cast<FloatType>(eltType))
      zeroElement = rewriter.getFloatAttr(flt, 0.0);
    else if (auto it = dyn_cast<IntegerType>(eltType))
      zeroElement = rewriter.getIntegerAttr(it, 0);
    auto zeroSplat = mlir::SplatElementsAttr::get(vecType, zeroElement);

    Value passthrough =
        rewriter.create<arith::ConstantOp>(loc, vecType, zeroSplat);
    return rewriter.create<vector::MaskedLoadOp>(loc, vecType, mem, indices,
                                                 mask, passthrough);
  } else {
    return rewriter.create<vector::LoadOp>(loc, vecType, mem, indices);
  }
}

static void buildVectorWrite(mlir::Location loc,
                             mlir::PatternRewriter &rewriter, mlir::Value mem,
                             llvm::ArrayRef<mlir::Value> indices,
                             mlir::Value vecValue,
                             std::optional<mlir::Value> maskOpt) {
  Value mask = maskOpt ? *maskOpt : mlir::Value{};
  if (mask) {
    rewriter.create<vector::MaskedStoreOp>(loc, mem, indices, mask, vecValue);
  } else {
    rewriter.create<vector::StoreOp>(loc, vecValue, mem, indices);
  }
}

class ReadOpLoweringPattern : public OpConversionPattern<wave::ReadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::ReadOp op, wave::ReadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    int elementsPerThread =
        32; // TODO: should not be hardcoded, get from somewhere

    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(op,
                                         "WaveTensorType conversion failed");

    auto vectorType = dyn_cast<VectorType>(convertedType);
    if (!vectorType)
      return rewriter.notifyMatchFailure(
          op, "expected vector type after conversion");

    Value base = adaptor.getMemory();
    auto memrefTy = dyn_cast<MemRefType>(base.getType());
    if (!memrefTy)
      return rewriter.notifyMatchFailure(
          op, "expected memref base after conversion");

    auto boundsAttr = op.getBoundsAttr();
    DictionaryAttr index = op.getIndexAttr();

    Operation *funcOp = op->getParentOfType<FunctionOpInterface>();
    wave::WaveHyperparameterAttr hyperparameter =
        funcOp->getAttrOfType<wave::WaveHyperparameterAttr>(
            wave::WaveDialect::kHyperparameterAttrName);

    // Build per-dimension start indices
    SmallVector<Value> indices =
        buildStartIndices(loc, index, rewriter, hyperparameter);

    auto mask = buildMask(loc, boundsAttr, rewriter, index, hyperparameter,
                          indices, elementsPerThread);

    Value ReadOp =
        buildVectorRead(loc, rewriter, base, indices, vectorType, mask);

    rewriter.replaceOp(op, ReadOp);

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
    auto memrefTy = dyn_cast<MemRefType>(base.getType());
    if (!memrefTy)
      return rewriter.notifyMatchFailure(op, "expected converted memref");

    Value vec = adaptor.getValueToStore();
    auto vecType = dyn_cast<VectorType>(vec.getType());
    if (!vecType)
      return rewriter.notifyMatchFailure(op, "expected vector value to store");

    int64_t elementsPerThread = vecType.getNumElements();
    auto boundsAttr = op.getBoundsAttr();
    DictionaryAttr index = op.getIndexAttr();

    Operation *funcOp = op->getParentOfType<mlir::FunctionOpInterface>();
    wave::WaveHyperparameterAttr hyperparameter =
        funcOp->getAttrOfType<wave::WaveHyperparameterAttr>(
            wave::WaveDialect::kHyperparameterAttrName);

    SmallVector<Value> indices =
        buildStartIndices(loc, index, rewriter, hyperparameter);

    // Build per-lane mask (or none)
    auto mask = buildMask(loc, boundsAttr, rewriter, index, hyperparameter,
                          indices, elementsPerThread);

    buildVectorWrite(loc, rewriter, base, indices, vec, mask);

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void wave::populateWaveReadWriteLoweringPatterns(
    WaveTensorTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ReadOpLoweringPattern, WriteOpLoweringPattern>(
      typeConverter, patterns.getContext());
}
