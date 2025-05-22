// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_PASSES
#define WATER_PASSES

include "mlir/Pass/PassBase.td"

def WaterAssertInBoundsPass : Pass<"water-assert-in-bounds"> {
  let summary = "Insert assertions that memref accesses are in bounds.";
  let dependentDialects = [
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "memref::MemRefDialect",
  ];
  let options = [
    Option<"warnOnUnknown", "warn-on-unknown", "bool", /*default=*/"true",
      "Warn on unknown side-effecting operations">,
    Option<"includeVectorLoadStore", "include-vector-load-store", "bool",
      /*default=*/"false",
      "Include vector.load/store operations despite them allowing "
      "out-of-bounds">,
    Option<"checkEachDim", "check-each-dim", "bool", /*default=*/"true",
      "Check each dimension individually">,
    // TODO: consider making this a separate pass as it has to operate at
    // module level.
    Option<"createSpeculativeFuncs", "create-speculative-funcs", "bool",
      /*default=*/"false",
      "Create a function that performs assertions speculatively instead of "
      "in-place checks">
  ];
}

def WaterCheckStaticAssertionsPass : Pass<"water-check-static-assertions"> {
  let summary = "Fail if the IR contains a known-false assertion.";
}

#endif // WATER_PASSES
