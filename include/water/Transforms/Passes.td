//===- Passes.td - Passes definitions ----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WATER_PASSES
#define WATER_PASSES

include "mlir/Pass/PassBase.td"

def WaterHelloWorldPass : Pass<"water-hello-world"> {
  let summary =
    "Prints a hello world as a note attached to the pass root operation";
}

#endif // WATER_PASSES
