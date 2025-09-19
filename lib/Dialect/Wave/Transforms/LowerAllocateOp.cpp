// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "water/Dialect/Wave/IR/WaveOps.h"

using namespace mlir;

namespace {

class AllocateOpLoweringPattern : public OpConversionPattern<wave::AllocateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::AllocateOp op, wave::AllocateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    wave::WaveTensorType resultType = op.getResult().getType();
    wave::DistributedShapeAttr distributedShape = op.getDistributedShape();
    auto *typeConverter =
        static_cast<const wave::WaveTypeConverter *>(getTypeConverter());
    Type convertedResultType = typeConverter->convertTensorFromComponents(
        distributedShape.getSymbolNames(), distributedShape.getShape(),
        resultType.getElementType(), resultType.getAddressSpaceValue());
    if (!convertedResultType) {
      return rewriter.notifyMatchFailure(op,
                                         "failed to construct resulting type");
    }

    auto memrefType = dyn_cast<MemRefType>(convertedResultType);
    if (!memrefType)
      return rewriter.notifyMatchFailure(
          op, "expected memref type for allocate op");
    Location loc = op.getLoc();

    // If operation contains a parent op, emit a view into this parent
    // allocation
    Value parent = adaptor.getParent();
    int64_t byteOffset = 0;
    if (parent) {
      // We need to ignore the automatically inserted cast. It is due to the
      // fact that the memref shape coming out of the allocation has a shape
      // different from one that would be obtained by converting the result
      // type... This is an annoying repercussion of the strange design choice
      // in original Wave of having "allocate" represent both allocations and
      // views, which makes it impossible to have a consistent result type: for
      // allocations, the shape is the distributed shape, but for views it is
      // the result shape.
      auto cast = parent.getDefiningOp<UnrealizedConversionCastOp>();
      assert(cast && "unexpected type conversion configuration");

      byteOffset = static_cast<int64_t>(*op.getOffset());
      mlir::Value off =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, byteOffset);
      rewriter.replaceOpWithNewOp<mlir::memref::ViewOp>(
          op, memrefType, cast.getOperand(0), off, mlir::ValueRange());

      return success();
    }

    // No parent : emit plain memref.alloc
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType);

    return success();
  }
};

} // namespace

void wave::populateWaveAllocateOpLoweringPatterns(
    WaveTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<AllocateOpLoweringPattern>(typeConverter, patterns.getContext());
}
