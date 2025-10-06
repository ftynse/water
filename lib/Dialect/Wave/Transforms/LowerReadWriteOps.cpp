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

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "wave-lowering"
#define LDBG() llvm::dbgs() << "[" DEBUG_TYPE "] "

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
      LLVM_DEBUG(llvm::errs() << "symbol: " << name << "\n");
      assert(false && "unknown symbol, should have been caught by verifiers");
    }
    baseSymVals.push_back(v);
  }
  // In case map contains multiple results, create one apply per result.
  SmallVector<Value> results;
  results.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    AffineMap submap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), expr);
    SmallVector<Value> symVals = baseSymVals;
    affine::canonicalizeMapAndOperands(&submap, &symVals);

    Value apply = rewriter.create<affine::AffineApplyOp>(loc, submap, symVals);
    results.push_back(apply);
  }

  return results;
}

/// Build per-dimension start indices in the order specified by `orderedSyms`
/// (the Wave tensor’s dim order). For each symbol name in `orderedSyms`, this
/// looks up a WaveIndexMappingAttr in `indexDict`, materializes its `start`
/// map and returns one index-typed Value per dimension.
static FailureOr<SmallVector<Value>>
buildStartIndices(Location loc, DictionaryAttr indexDict,
                  ArrayRef<wave::WaveSymbolAttr> orderedSyms,
                  PatternRewriter &rewriter,
                  wave::WaveHyperparameterAttr hyper) {
  SmallVector<Value> indices;
  indices.reserve(orderedSyms.size());
  for (wave::WaveSymbolAttr symAttr : orderedSyms) {
    StringRef name = symAttr.getName();
    Attribute a = indexDict.get(name);
    assert(a && "index dict missing entry for dimension symbol");
    auto mapAttr = cast<wave::WaveIndexMappingAttr>(a);

    FailureOr<SmallVector<Value>> startFo = materializeAffine(
        loc, mapAttr.getSymbolNames(), mapAttr.getStart(), rewriter, hyper);
    if (failed(startFo))
      return failure();
    SmallVector<Value> start = std::move(*startFo);
    assert(llvm::hasSingleElement(start));
    indices.push_back(start[0]); // Start map has one result.
  }
  return indices;
}

/// Build a per-thread mask:
///
///   mask = AND_d ( id_start_d(elements_per_thread) <
///                  bound_d(elements_per_thread))
///          foreach d in dimensions.
///
/// whenever a bounds dictionary is provided. When it is not provided, return a
/// null mask. If the vectorized dimension cannot be identified, return failure.
static FailureOr<Value>
buildMask(Location loc, wave::WaveReadWriteBoundsAttr boundsDict,
          ArrayRef<wave::WaveSymbolAttr> orderedSyms, PatternRewriter &rewriter,
          DictionaryAttr indexDict, wave::WaveHyperparameterAttr hyper,
          ArrayRef<Value> startIdx, int64_t elementsPerThread) {
  if (!boundsDict)
    return Value();

  const uint64_t rank = startIdx.size();
  IndexType idxType = rewriter.getIndexType();
  VectorType vecIdxType = VectorType::get({elementsPerThread}, idxType);
  IntegerType i1Type = IntegerType::get(rewriter.getContext(), 1);
  VectorType maskType = VectorType::get({elementsPerThread}, i1Type);

  // finalMask is the AND of per-dimension bound checks.
  Value finalMask;
  for (uint64_t d = 0; d < rank; ++d) {
    StringRef name = orderedSyms[d].getName();
    Attribute a = boundsDict.getMapping().get(name);
    assert(a && "bounds dict missing entry for dimension symbol");
    auto boundAttr = cast<wave::ExprAttr>(a);
    // Materialize bounds.
    FailureOr<SmallVector<Value>> boundValsFo = materializeAffine(
        loc, boundAttr.getSymbolNames(), boundAttr.getShape(), rewriter, hyper);
    if (failed(boundValsFo))
      return failure();
    SmallVector<Value> boundVals = std::move(*boundValsFo);
    Value bound = boundVals[0];

    Value clause;
    if (d == rank - 1) {
      // iota [0..L-1] : vector<index>
      Value iota = rewriter.create<vector::StepOp>(loc, vecIdxType);

      // Lane indices for fastest dim: start + iota.
      Value startFastVec =
          rewriter.create<vector::BroadcastOp>(loc, vecIdxType, startIdx[d]);
      Value laneIdxFast =
          rewriter.create<arith::AddIOp>(loc, startFastVec, iota);

      // lane-wise compare: (start + iota) < bound.
      Value boundVec =
          rewriter.create<vector::BroadcastOp>(loc, vecIdxType, bound);
      clause = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                              laneIdxFast, boundVec);
    } else {
      // scalar compare then broadcast: startIdx[d] < bound.
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

/// Build a read or a mask read operation based on presence of a mask.
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

/// Build a write or a masked write operation based on presence of a mask.
static void buildVectorWrite(Location loc, PatternRewriter &rewriter, Value mem,
                             ArrayRef<Value> indices, Value vecValue,
                             Value mask) {
  if (mask) {
    rewriter.create<vector::MaskedStoreOp>(loc, mem, indices, mask, vecValue);
  } else {
    rewriter.create<vector::StoreOp>(loc, vecValue, mem, indices);
  }
}

// Common lowering for memory operations such as read/write. Checks whether
// enough information is present on the operation and constructs the starting
// index (populates the trailing vector argument) and eventually the mask.
template <typename OpTy>
static FailureOr<Value> createMemoryIndicesAndMask(
    ConversionPatternRewriter &rewriter, const TypeConverter *typeConverter,
    OpTy op, wave::WaveTensorType memoryType, VectorType vectorType,
    SmallVectorImpl<Value> &startIndices) {

  int64_t elementsPerThread = vectorType.getNumElements();

  ArrayRef<wave::WaveSymbolAttr> orderedSyms = memoryType.getShape();

  wave::WaveReadWriteBoundsAttr boundsDict = op.getBoundsAttr();
  wave::WaveHyperparameterAttr hyper =
      static_cast<const wave::WaveTypeConverter &>(*typeConverter)
          .getHyperparameters();

  // This logic operates after the "expansion" pass that usually unrolls all
  // tensor dimensions but one so the extent along them is 1. The dimension
  // with non-unit extent is considered to be vectorized.
  DictionaryAttr indexDict = op.getIndexAttr();
  if (!indexDict)
    return rewriter.notifyMatchFailure(
        op, "cannot lower without 'index' attribute");
  std::optional<uint64_t> vectorizedDim =
      wave::getPositionOfVectorizedDim(orderedSyms, indexDict, hyper);

  if (!vectorizedDim.has_value() ||
      *vectorizedDim != memoryType.getRank() - 1) {
    return rewriter.notifyMatchFailure(
        op, "unsupported load along non-innermost dimension");
  }

  FailureOr<SmallVector<Value>> maybeStartIndices =
      buildStartIndices(op->getLoc(), indexDict, orderedSyms, rewriter, hyper);
  if (failed(maybeStartIndices))
    return rewriter.notifyMatchFailure(
        op, "failed to convert start indices to affine");
  startIndices = std::move(*maybeStartIndices);

  FailureOr<Value> mask =
      buildMask(op->getLoc(), boundsDict, orderedSyms, rewriter, indexDict,
                hyper, startIndices, elementsPerThread);
  if (failed(mask))
    return rewriter.notifyMatchFailure(op, "couldn't build the required mask");

  return mask;
}

class ReadOpLoweringPattern : public OpConversionPattern<wave::ReadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::ReadOp op, wave::ReadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(op,
                                         "WaveTensorType conversion failed");
    auto vectorType = cast<VectorType>(convertedType);
    SmallVector<Value> startIndices;
    FailureOr<Value> mask = createMemoryIndicesAndMask(
        rewriter, getTypeConverter(), op, op.getMemory().getType(), vectorType,
        startIndices);
    if (failed(mask))
      return failure();

    Value readOp = buildVectorRead(op.getLoc(), rewriter, adaptor.getMemory(),
                                   startIndices, vectorType, *mask);
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
    Value vec = adaptor.getValueToStore();
    auto vecType = cast<VectorType>(vec.getType());
    SmallVector<Value> startIndices;
    FailureOr<Value> mask = createMemoryIndicesAndMask(
        rewriter, getTypeConverter(), op, op.getMemory().getType(), vecType,
        startIndices);
    if (failed(mask))
      return failure();

    buildVectorWrite(op.getLoc(), rewriter, adaptor.getMemory(), startIndices,
                     vec, *mask);
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
