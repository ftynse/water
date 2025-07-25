//===- water-opt.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "water/Tools/water-opt/WaterOptMain.h"
#include "water/Transforms/Passes.h"

// Forward-declare test passes so we don't have a dependency on the test
// headers.
namespace mlir::water::test {
void registerAllPasses();
} // namespace mlir::water::test

int main(int argc, char **argv) {
  mlir::water::registerPasses();
  mlir::water::test::registerAllPasses();

  mlir::arith::registerArithIntRangeOptsPass();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerLowerAffinePass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::amdgpu::AMDGPUDialect,
                  mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::gpu::GPUDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::vector::VectorDialect>();

  return mlir::asMainReturnCode(
      WaterOptMain(argc, argv, "water optimizer driver\n", registry));
}
