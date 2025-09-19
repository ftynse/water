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

/// Lowers `wave.register` into a constant vector to avoid memory allocation.
class RegisterOpLoweringPattern : public OpConversionPattern<wave::RegisterOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(wave::RegisterOp op, wave::RegisterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = getTypeConverter()->convertType(op);
    if (!convertedType) {
      return rewriter.notifyMatchFailure(op,
                                         "WaveTensorType conversion failed");
    }
    auto vectorType = dyn_cast<VectorType>(convertedType);
    if (!vectorType) {
      return rewriter.notifyMatchFailure(
          op, "expected vector type after conversion");
    }

    TypedAttr splatAttr;
    Value initValue = op.getInit();
    Type elementType = vectorType.getElementType();

    if (isa<FloatType>(elementType)) {
      if (auto cst = initValue.getDefiningOp<arith::ConstantFloatOp>()) {
        splatAttr = DenseFPElementsAttr::get(vectorType, cst.value());
      } else {
        return rewriter.notifyMatchFailure(op,
                                           "init value must be constant float");
      }
    } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
      if (auto cst = initValue.getDefiningOp<arith::ConstantIntOp>()) {
        if (!intType.isSignless()) {
          return rewriter.notifyMatchFailure(
              op, "only signless integer types are supported");
        }
        APInt valueInt(elementType.getIntOrFloatBitWidth(), cst.value(),
                       /*isSigned=*/true);
        splatAttr = DenseIntElementsAttr::get(vectorType, valueInt);
      } else {
        return rewriter.notifyMatchFailure(
            op, "init value must be constant integer");
      }
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    // Create vector.constant operation
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, vectorType, splatAttr);

    return success();
  }
};

} // namespace

void wave::populateWaveRegisterLoweringPatterns(
    WaveTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<RegisterOpLoweringPattern>(typeConverter, patterns.getContext());
}
