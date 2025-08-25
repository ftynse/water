// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_IR_WAVEAINTERFACES_H
#define WATER_DIALECT_WAVE_IR_WAVEAINTERFACES_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "water/Dialect/Wave/IR/WaveOpInterfaces.h.inc"
#include "llvm/Support/raw_ostream.h"

namespace wave {

namespace detail {
llvm::FailureOr<mlir::ChangeResult>
identityTypeInferencePropagate(llvm::ArrayRef<wave::WaveTensorType> from,
                               llvm::MutableArrayRef<wave::WaveTensorType> to,
                               llvm::StringRef fromName, llvm::StringRef toName,
                               llvm::raw_ostream &errs);
} // namespace detail

template <typename OpTy>
class IdentityTypeInferenceOpTrait
    : public mlir::OpTrait::TraitBase<OpTy, IdentityTypeInferenceOpTrait> {
public:
  llvm::FailureOr<mlir::ChangeResult>
  propagateForward(llvm::ArrayRef<wave::WaveTensorType> operandTypes,
                   llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
                   llvm::raw_ostream &errs) {
    return wave::detail::identityTypeInferencePropagate(
        operandTypes, resultTypes, "operands", "results", errs);
  }

  llvm::FailureOr<mlir::ChangeResult>
  propagateBackward(llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
                    llvm::ArrayRef<wave::WaveTensorType> resultTypes,
                    llvm::raw_ostream &errs) {
    return wave::detail::identityTypeInferencePropagate(
        resultTypes, operandTypes, "results", "operands", errs);
  }
};
} // namespace wave

#endif // WATER_DIALECT_WAVE_IR_WAVEAINTERFACES_H
