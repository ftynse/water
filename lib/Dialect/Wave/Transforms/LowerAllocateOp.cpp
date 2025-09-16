
// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"

#include "mlir/Dialect/MemRef/IR/Memref.h"
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
    Type convertedType = getTypeConverter()->convertType(op);

    if (!convertedType) {
      return rewriter.notifyMatchFailure(op,
                                         "WaveTensorType conversion failed");
    }
    auto memrefType = dyn_cast<MemRefType>(convertedType);
    if (!memrefType)
      return rewriter.notifyMatchFailure(
          op, "expected memref type for allocate op");
    Location loc = op.getLoc();

    // If operation contains a parent op, emit a view into this parent
    // allocation
    Value parent = adaptor.getParent();
    int64_t byteOffset = 0;
    if (parent) {
      byteOffset = static_cast<int64_t>(*op.getOffset());
      mlir::Value off =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, byteOffset);
      rewriter.replaceOpWithNewOp<mlir::memref::ViewOp>(
          op, memrefType, parent, off, mlir::ValueRange());

      return success();
    }

    // No parent : emit plain memref.alloc
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType);

    return success();
  }
};

} // namespace

void wave::populateWaveAllocateOpLoweringPatterns(
    WaveTensorTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<AllocateOpLoweringPattern>(typeConverter, patterns.getContext());
}
