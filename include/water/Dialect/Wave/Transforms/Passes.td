// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_TRANSFORMS_PASSES
#define WATER_DIALECT_WAVE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LowerWaveToMLIRPass : Pass<"lower-wave-to-mlir", "mlir::ModuleOp"> {
  let summary = "Lower Wave dialect to upstream MLIR dialects";
  let description = [{
    This pass lowers operations from the Wave dialect to other upstream MLIR
    dialects, notable vector/memref.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::vector::VectorDialect"
  ];
}

#endif // WATER_DIALECT_WAVE_TRANSFORMS_PASSES
