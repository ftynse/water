// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"

#include "mlir/IR/BuiltinTypes.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

mlir::water::WaveTensorTypeConverter::WaveTensorTypeConverter() {
  addConversion([](wave::WaveTensorType type) -> std::optional<Type> {
    auto shape = type.getResolvedShape();
    // Fail if shapes aren't resolved
    if (shape.empty())
      return std::nullopt;
    // Convert WaveTensorInRegister to VectorType, and WaveTensorInMemory to
    // MemRefType
    auto addrSpace = type.getAddressSpaceValue();
    if (addrSpace == wave::WaveAddressSpace::Register)
      return VectorType::get(shape, type.getElementType());
    // TODO: add gpu memory space
    return MemRefType::get(shape, type.getElementType());
  });

  // Mark all other types as legal
  addConversion([](Type type) -> std::optional<Type> {
    if (!isa<wave::WaveTensorType>(type))
      return type;
    return std::nullopt;
  });

  addSourceMaterialization([](OpBuilder &builder, wave::WaveTensorType waveType,
                              ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    return UnrealizedConversionCastOp::create(builder, loc, waveType, inputs)
        .getResult(0);
  });

  addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  });
}
