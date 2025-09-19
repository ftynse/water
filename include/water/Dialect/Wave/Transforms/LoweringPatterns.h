// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_TRANSFORMS_TRANSFORMS_H
#define WATER_DIALECT_WAVE_TRANSFORMS_TRANSFORMS_H

#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace wave {

enum class WaveAddressSpace : uint32_t;
class WaveHyperparameterAttr;
class WaveSymbolAttr;

struct WaveTensorTypeConverter : public mlir::TypeConverter {
  WaveTensorTypeConverter();

  mlir::Type convertTensorFromComponents(
      llvm::ArrayRef<wave::WaveSymbolAttr> symbols, mlir::AffineMap shape,
      mlir::Type elementType, wave::WaveAddressSpace addressSpace,
      wave::WaveHyperparameterAttr hyperParameters) const;

  wave::WaveHyperparameterAttr getHyperparametersAt(mlir::Value value) const;
};

// Adds pattern that lowers `wave.register` to upstream MLIR ops.
void populateWaveRegisterLoweringPatterns(
    WaveTensorTypeConverter &typeConverter, mlir::RewritePatternSet &patterns);

// Adds pattern that lowers wave binary ops to upstream MLIR ops.
void populateWaveBinaryOpLoweringPatterns(
    WaveTensorTypeConverter &typeConverter, mlir::RewritePatternSet &patterns);

// Adds pattern that lowers 'wave.allocate' ops to upstream MLIR ops.
void populateWaveAllocateOpLoweringPatterns(
    WaveTensorTypeConverter &typeConverter, mlir::RewritePatternSet &patterns);

} // namespace wave

#endif // WATER_DIALECT_WAVE_TRANSFORMS_TRANSFORMS_H
