
// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"

#include "mlir/Dialect/Memref/IR/Memref.h"
#include "mlir/Transforms/DialectConversion.h"
#include "water/Dialect/Wave/IR/WaveOps.h"

using namespace mlir;

namespace {

/// Lowers allocate ops into either plain memref.alloc, but if you have a parent
/// buffer or tail padding, it lowers to a combination (e.g., memref.alloc +
/// memref.view)

class AllocateOpLoweringPattern : public OpConversionPattern<wave::AllocateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::AllocateOp op, wave::AllocateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResTy =
        getTypeConverter()->convertType(op.getResult().getType());
    auto memrefTy = dyn_cast<MemRefType>(convertedResTy);
    if (!memrefTy)
      return rewriter.notifyMatchFailure(
          op, "result type did not convert to memref");

    // TODO: take into account parent op and padding

    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefTy);

    return success();
  }
};

} // namespace

void wave::populateWaveAllocateOpLoweringPatterns(
    WaveTensorTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<AllocateOpLoweringPattern>(typeConverter, patterns.getContext());
}
