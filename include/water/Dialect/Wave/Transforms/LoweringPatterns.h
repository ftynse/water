// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_TRANSFORMS_TRANSFORMS_H
#define WATER_DIALECT_WAVE_TRANSFORMS_TRANSFORMS_H

#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class RewritePatternSet;

namespace water {

struct WaveTensorTypeConverter : public TypeConverter {
  WaveTensorTypeConverter();
};

/// Lowers `wave.register` into a sequence of division and remainder
/// operations.
void populateWaveRegisterLoweringPatterns(
    WaveTensorTypeConverter &typeConverter, RewritePatternSet &patterns);

} // namespace water
} // namespace mlir

#endif // WATER_DIALECT_WAVE_TRANSFORMS_TRANSFORMS_H
