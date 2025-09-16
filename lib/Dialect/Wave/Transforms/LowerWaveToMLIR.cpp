// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"
#include "water/Dialect/Wave/Transforms/Utils.h"

#define GEN_PASS_DEF_LOWERWAVETOMLIRPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"

using namespace mlir;

namespace {

struct LowerWaveToMLIRPass
    : public ::impl::LowerWaveToMLIRPassBase<LowerWaveToMLIRPass> {
  using LowerWaveToMLIRPassBase::LowerWaveToMLIRPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();

    // TODO: require index expressions to be present
    if (failed(wave::verifyNormalFormPassPrecondition(
            wave::WaveNormalForm::AllTypesSpecified, op, getPassName())))
      return signalPassFailure();

    wave::WaveTensorTypeConverter typeConverter;
    ConversionTarget target(*ctx);

    target.addLegalDialect<arith::ArithDialect, vector::VectorDialect,
                           memref::MemRefDialect>();
    target.addIllegalOp<wave::RegisterOp, wave::AllocateOp>();

    RewritePatternSet patterns(ctx);
    wave::populateWaveRegisterLoweringPatterns(typeConverter, patterns);
    wave::populateWaveBinaryOpLoweringPatterns(typeConverter, patterns);
    wave::populateWaveAllocateOpLoweringPatterns(typeConverter, patterns);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(
            applyPartialConversion(op, target, std::move(patterns), config))) {
      return signalPassFailure();
    }

    if (failed(wave::setNormalFormPassPostcondition(wave::WaveNormalForm::None,
                                                    op)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> wave::createLowerWaveToMLIRPass() {
  return std::make_unique<LowerWaveToMLIRPass>();
}
