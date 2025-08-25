// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveInterfaces.h"

#include "water/Dialect/Wave/IR/WaveOpInterfaces.cpp.inc"
#include "water/Dialect/Wave/IR/WaveTypes.h"

static bool hasSameShape(wave::WaveTensorType lhs, wave::WaveTensorType rhs) {
  // TODO: this may require more advanced checking if shapes are more complex
  // than a single symbol.
  return lhs && rhs && lhs.getShape() == rhs.getShape();
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
    if (!*it)
      continue;

    if (hasSameShape(fr, *it))
      continue;

    errs << "irreconcilable types (" << fr << " vs " << *it
         << ") during type inference from " << fromName << " to " << toName
         << " for type #" << i;
    return mlir::failure();
  }

  mlir::ChangeResult changeResult = mlir::ChangeResult::NoChange;
  for (auto &&[i, toType] : llvm::enumerate(to)) {
    if (!toType)
      continue;

    if (!toType.getFullySpecified()) {
      changeResult = mlir::ChangeResult::Change;
      toType = *it;
      continue;
    }
    if (toType == *it)
      continue;

    errs << "propagating type shapes from " << fromName << " to " << toName
         << " results in a conflict (existing type " << toType
         << ", propagated " << *it << ") for " << fromName << " #" << i;
    return mlir::failure();
  }
  return changeResult;
}
