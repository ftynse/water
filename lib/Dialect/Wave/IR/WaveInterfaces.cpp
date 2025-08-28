// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveInterfaces.h"

#include "water/Dialect/Wave/IR/WaveOpInterfaces.cpp.inc"
#include "water/Dialect/Wave/IR/WaveTypes.h"

// Return `true` if two tensor types have the same shape. Null types are
// considered to have different shapes.
static bool hasSameShape(wave::WaveTensorType lhs, wave::WaveTensorType rhs) {
  // TODO: this may require more advanced checking if shapes are more complex
  // than a single symbol.
  return lhs && rhs && lhs.getShape() == rhs.getShape();
}

llvm::FailureOr<mlir::ChangeResult> wave::detail::checkPropagateShapeConflict(
    wave::WaveTensorType from, wave::WaveTensorType to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  if (!from || !to || hasSameShape(from, to))
    return mlir::ChangeResult::NoChange;

  if (!to.getFullySpecified())
    return mlir::ChangeResult::Change;

  errs << "irreconcilable types during type inference from " << fromName << "("
       << from << ") to " << toName << "(" << to << ")";
  return mlir::failure();
}

llvm::FailureOr<mlir::ChangeResult> wave::detail::propagateShapeInformation(
    wave::WaveTensorType from, wave::WaveTensorType &to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  llvm::FailureOr<mlir::ChangeResult> res =
      checkPropagateShapeConflict(from, to, fromName, toName, errs);
  if (mlir::failed(res) || *res == mlir::ChangeResult::NoChange)
    return res;

  to = to.copyShapeFrom(from);
  return mlir::ChangeResult::Change;
}

llvm::FailureOr<mlir::ChangeResult>
wave::detail::identityTypeInferencePropagate(
    llvm::ArrayRef<wave::WaveTensorType> from,
    llvm::MutableArrayRef<wave::WaveTensorType> to, llvm::StringRef fromName,
    llvm::StringRef toName, llvm::raw_ostream &errs) {
  auto it = llvm::find_if(from, [](wave::WaveTensorType type) {
    return type && type.getFullySpecified();
  });
  if (it == from.end())
    return mlir::ChangeResult::NoChange;

  // Expect all fully-specified "from" types to have the same shape.
  for (auto [i, fr] : llvm::enumerate(from)) {
    llvm::FailureOr<mlir::ChangeResult> res =
        checkPropagateShapeConflict(*it, fr, fromName, toName, errs);
    if (mlir::failed(res)) {
      errs << " for " << fromName << " #" << i;
      return res;
    }
  }

  mlir::ChangeResult changeResult = mlir::ChangeResult::NoChange;
  for (auto &&[i, toType] : llvm::enumerate(to)) {
    llvm::FailureOr<mlir::ChangeResult> res =
        propagateShapeInformation(*it, toType, fromName, toName, errs);
    if (mlir::failed(res)) {
      errs << " for " << fromName << " #" << i;
      return mlir::failure();
    }

    changeResult |= *res;
  }
  return changeResult;
}
