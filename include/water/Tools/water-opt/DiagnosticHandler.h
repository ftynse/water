//===- DiagnosticHandler.h - Water Diagnostics -------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for emitting diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef WATER_TOOLS_WATEROPT_DIAGNOSTICSHANDLER_H
#define WATER_TOOLS_WATEROPT_DIAGNOSTICSHANDLER_H

#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

class WaterDiagnosticHandler : public ScopedDiagnosticHandler {
public:
  WaterDiagnosticHandler(MLIRContext *ctx, llvm::raw_ostream &os)
      : ScopedDiagnosticHandler(ctx) {
    setHandler([&os](Diagnostic &diag) {
      // TODO: properly serialize
      os << diag << "\n";
      return failure();
    });
  }
};

}; // namespace mlir

#endif // WATER_TOOLS_WATEROPT_DIAGNOSTICSHANDLER_H
