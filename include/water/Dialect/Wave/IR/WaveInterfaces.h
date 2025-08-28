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
// Propagate shape information from `from` tensor types to `to` tensor types.
// Expects all fully-specified tensor types to have the same shape, prints an
// error message to `errs` otherwise. Update all under-specified tensor types in
// `to` to be fully specified with the shapes extracted from from. For
// fully-specified types in `to`, check if their shapes match those in `from`
// and print error messages to `errs` otherwise. The error message uses `toName`
// and `fromName` to to describe `from` and `to` tensors. If there was an error
// due to mismatching/irreconcilable types and an error was printed, returns
// `failure`. Otherwise returns an change indicator.
llvm::FailureOr<mlir::ChangeResult>
identityTypeInferencePropagate(llvm::ArrayRef<wave::WaveTensorType> from,
                               llvm::MutableArrayRef<wave::WaveTensorType> to,
                               llvm::StringRef fromName, llvm::StringRef toName,
                               llvm::raw_ostream &errs);
llvm::FailureOr<mlir::ChangeResult>
propagateShapeInformation(wave::WaveTensorType from, wave::WaveTensorType &to,
                          llvm::StringRef fromName, llvm::StringRef toName,
                          llvm::raw_ostream &errs);

// Check whether the `from` and `to` tensor types have reconcilable shapes and
// and print error messages to `errs` otherwise. The error message uses `toName`
// and `fromName` to to describe `from` and `to` tensors. If types are
// reconcilable, returns an indicator whether the `to` type will have to be
// updated.
llvm::FailureOr<mlir::ChangeResult>
checkPropagateShapeConflict(wave::WaveTensorType from, wave::WaveTensorType to,
                            llvm::StringRef fromName, llvm::StringRef toName,
                            llvm::raw_ostream &errs);

} // namespace detail

// A trait providing an implementation of the WaveInferTypeOpInterface where
// shapes are propagated from all operands to all results and back as is.
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

// A trait providing an implementation of the WaveInferTypeOpInterface where no
// shape propagation is needed. E.g. for operations that only have operands and
// no results.
template <typename OpTy>
class NoOpTypeInferenceOpTrait
    : public mlir::OpTrait::TraitBase<OpTy, NoOpTypeInferenceOpTrait> {
public:
  llvm::FailureOr<mlir::ChangeResult>
  propagateForward(llvm::ArrayRef<wave::WaveTensorType> operandTypes,
                   llvm::MutableArrayRef<wave::WaveTensorType> resultTypes,
                   llvm::raw_ostream &errs) {
    return mlir::ChangeResult::NoChange;
  }

  llvm::FailureOr<mlir::ChangeResult>
  propagateBackward(llvm::MutableArrayRef<wave::WaveTensorType> operandTypes,
                    llvm::ArrayRef<wave::WaveTensorType> resultTypes,
                    llvm::raw_ostream &errs) {
    return mlir::ChangeResult::NoChange;
  }
};
} // namespace wave

#endif // WATER_DIALECT_WAVE_IR_WAVEAINTERFACES_H
