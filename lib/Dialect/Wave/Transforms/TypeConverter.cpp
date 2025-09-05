// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "wave-tensor-type-converter"

using namespace mlir;

wave::WaveTensorTypeConverter::WaveTensorTypeConverter() {
  addConversion([](Type type) -> std::optional<Type> {
    if (auto waveType = dyn_cast<wave::WaveTensorType>(type)) {
      auto shape = waveType.getResolvedShape();
      // Fail if shapes aren't resolved.
      if (shape.empty()) {
        LLVM_DEBUG({
          llvm::dbgs() << "WaveTensorType conversion failed: symbolic shape "
                          "unresolved\n";
        });
        return std::nullopt;
      }
      // Convert WaveTensorInRegister to VectorType, and WaveTensorInMemory to
      // MemRefType with proper memory space.
      wave::WaveAddressSpace addrSpace = waveType.getAddressSpaceValue();
      Type elementType = waveType.getElementType();

      switch (addrSpace) {
      case wave::WaveAddressSpace::Unspecified:
        return VectorType::get(shape, elementType);

      case wave::WaveAddressSpace::Global: {
        // GPU global memory (device memory)
        auto globalMemoryAddressSpace = gpu::AddressSpaceAttr::get(
            elementType.getContext(), gpu::AddressSpace::Global);
        return MemRefType::get(shape, elementType,
                               /*layout=*/MemRefLayoutAttrInterface{},
                               Attribute(globalMemoryAddressSpace));
      }

      case wave::WaveAddressSpace::Shared: {
        // GPU shared memory
        auto workgroupMemoryAddressSpace = gpu::AddressSpaceAttr::get(
            elementType.getContext(), gpu::AddressSpace::Workgroup);
        return MemRefType::get(shape, elementType,
                               /*layout=*/MemRefLayoutAttrInterface{},
                               Attribute(workgroupMemoryAddressSpace));
      }

      case wave::WaveAddressSpace::Register:
        // For register space, use vector type (registers are handled by LLVM)
        return VectorType::get(shape, elementType);

      default:
        LLVM_DEBUG({
          llvm::dbgs() << "Unknown WaveAddressSpace: "
                       << static_cast<int>(addrSpace) << "\n";
        });
        return std::nullopt;
      }
    }
    // Mark all other types as legal.
    return type;
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
