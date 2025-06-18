//===- Passes.cpp - Pass definitions ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "water/Transforms/Passes.h"

namespace mlir::water {
#define GEN_PASS_DEF_WATERHELLOWORLDPASS
#include "water/Transforms/Passes.h.inc"

namespace {
class HelloWorldPass : public impl::WaterHelloWorldPassBase<HelloWorldPass> {
public:
  void runOnOperation() final { getOperation()->emitRemark("Hello, world!"); }
};
} // namespace
} // namespace mlir::water
