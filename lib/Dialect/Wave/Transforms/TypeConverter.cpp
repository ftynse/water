// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/LoweringPatterns.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "wave-tensor-type-converter"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;

/// Gets the nearest enclosing function operation for a given SSA value.
static Operation *getEnclosingFunction(Value v) {
  if (Operation *definingOp = v.getDefiningOp())
    return definingOp->getParentOfType<FunctionOpInterface>();

  auto blockArg = cast<BlockArgument>(v);
  Block *block = blockArg.getOwner();
  Region *region = block->getParent();
  if (!region)
    return nullptr;

  return region->getParentOp()->getParentOfType<FunctionOpInterface>();
}

wave::WaveTensorTypeConverter::WaveTensorTypeConverter() {
  addConversion([](Value v) -> std::optional<Type> {
    Type type = v.getType();
    Operation *funcOp = getEnclosingFunction(v);
    if (!funcOp) {
      return std::nullopt;
    }
    auto hyperparameterAttr = funcOp->getAttrOfType<WaveHyperparameterAttr>(
        WaveDialect::kHyperparameterAttrName);
    if (!hyperparameterAttr)
      return std::nullopt;

    if (auto waveType = dyn_cast<wave::WaveTensorType>(type)) {

      std::optional<llvm::SmallVector<int64_t>> shape;

      // If this value comes from `wave.allocate`, resolve its distributed
      // (physical) shape
      if (Operation *def = v.getDefiningOp()) {
        if (auto alloc = llvm::dyn_cast<wave::AllocateOp>(def)) {
          wave::DistributedShapeAttr dist = alloc.getDistributedShape();
          shape = dist.getResolvedShape(hyperparameterAttr);
        }
      }
      // Fallback: resolve from the WaveTensorType's own (symbolic) shape
      if (!shape) {
        shape = waveType.getResolvedShape(hyperparameterAttr);
        // Fail if shapes aren't resolved.
        if (shape == std::nullopt) {
          LLVM_DEBUG({
            DBGS() << "WaveTensorType conversion failed: symbolic shape "
                      "unresolved\n";
          });
          return std::nullopt;
        }
      }
      // Convert WaveTensorInRegister to VectorType, and WaveTensorInMemory to
      // MemRefType with proper memory space.
      wave::WaveAddressSpace addrSpace = waveType.getAddressSpaceValue();
      Type elementType = waveType.getElementType();

      switch (addrSpace) {
      case wave::WaveAddressSpace::Unspecified:
        LLVM_DEBUG(DBGS() << "address spaces must have been specified\n");
        return std::nullopt;

      case wave::WaveAddressSpace::Global: {
        // GPU global memory (device memory)
        auto globalMemoryAddressSpace = gpu::AddressSpaceAttr::get(
            elementType.getContext(), gpu::AddressSpace::Global);
        return MemRefType::get(*shape, elementType,
                               /*layout=*/MemRefLayoutAttrInterface{},
                               Attribute(globalMemoryAddressSpace));
      }

      case wave::WaveAddressSpace::Shared: {
        // GPU shared memory
        auto workgroupMemoryAddressSpace = gpu::AddressSpaceAttr::get(
            elementType.getContext(), gpu::AddressSpace::Workgroup);
        return MemRefType::get(*shape, elementType,
                               /*layout=*/MemRefLayoutAttrInterface{},
                               Attribute(workgroupMemoryAddressSpace));
      }

      case wave::WaveAddressSpace::Register:
        // For register space, use vector type (registers are handled by LLVM)
        return VectorType::get(*shape, elementType);
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
