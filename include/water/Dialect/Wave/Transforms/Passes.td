// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_TRANSFORMS_PASSES
#define WATER_DIALECT_WAVE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def WaterWaveInferTypesPass : Pass<"water-wave-infer-types"> {
  let summary = "Infer fully-specified Wave dialect types";
}

#endif // WATER_DIALECT_WAVE_TRANSFORMS_PASSES
