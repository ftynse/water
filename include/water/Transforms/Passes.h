//===- Passes.h - Pass declarations  ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef WATER_TRANSFORMS_PASSES_H
#define WATER_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace water {
#define GEN_PASS_DECL
#include "water/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "water/Transforms/Passes.h.inc"
} // namespace water
} // namespace mlir

#endif
