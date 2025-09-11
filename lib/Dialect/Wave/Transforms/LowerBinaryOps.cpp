// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include "water/Dialect/Wave/IR/WaveOps.h"

using namespace mlir;

namespace {

/// Lowers binary ops into arith counterpart.
template <typename WaveOp, typename FloatOp, typename IntOp>
class BinaryOpLoweringPattern : public OpConversionPattern<WaveOp> {
public:
  using OpConversionPattern<WaveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaveOp op, typename WaveOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = this->getTypeConverter()->convertType(op);
    if (!convertedType) {
      return rewriter.notifyMatchFailure(op,
                                         "WaveTensorType conversion failed");
    }
    auto vectorType = dyn_cast<VectorType>(convertedType);
    if (!vectorType) {
      return rewriter.notifyMatchFailure(op,
                                         "expected vector type for binary op");
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Type elementType = vectorType.getElementType();

    if (isa<FloatType>(elementType)) {
      rewriter.replaceOpWithNewOp<FloatOp>(op, vectorType, lhs, rhs);
    } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
      rewriter.replaceOpWithNewOp<IntOp>(op, vectorType, lhs, rhs);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    return success();
  }
};

} // namespace

void wave::populateWaveBinaryOpLoweringPatterns(
    WaveTensorTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns
      .add<BinaryOpLoweringPattern<wave::AddOp, arith::AddFOp, arith::AddIOp>,
           BinaryOpLoweringPattern<wave::MulOp, arith::MulFOp, arith::MulIOp>,
           BinaryOpLoweringPattern<wave::DivOp, arith::DivFOp, arith::DivSIOp>>(
          typeConverter, patterns.getContext());
}
