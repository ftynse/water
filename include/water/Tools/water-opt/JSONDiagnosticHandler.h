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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

class JSONDiagnosticHandler : public ScopedDiagnosticHandler {
public:
  JSONDiagnosticHandler(MLIRContext *ctx, llvm::raw_ostream &os)
      : ScopedDiagnosticHandler(ctx) {
    setHandler([&](Diagnostic &diag) {
      Location loc = diag.getLocation();
      auto fileLoc = dyn_cast<FileLineColLoc>(loc);

      if (!fileLoc)
        return failure();

      StringRef file = fileLoc.getFilename().strref();
      unsigned line = fileLoc.getLine();
      unsigned col = fileLoc.getColumn();
      std::string msg = diag.str();

      llvm::json::Value json = llvm::json::Object{
          {"file", file}, {"line", line}, {"column", col}, {"message", msg}};

      os << json << "\n";

      return failure();
    });
  }
};

}; // namespace mlir

#endif // WATER_TOOLS_WATEROPT_DIAGNOSTICSHANDLER_H
