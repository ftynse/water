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
#include "mlir/IR/Location.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

/// Attempt to extract a filename for the given loc.
static std::optional<FileLineColLoc> extractFileLoc(Location loc) {
  while (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc))
    loc = callSiteLoc.getCallee();

  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return extractFileLoc(nameLoc.getChildLoc());
  if (auto opaqueLoc = dyn_cast<OpaqueLoc>(loc))
    return extractFileLoc(opaqueLoc.getFallbackLocation());
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (auto loc : fusedLoc.getLocations()) {
      if (auto fileLoc = extractFileLoc(loc))
        return fileLoc;
    }
  }
  return {};
}

class JSONDiagnosticHandler : public ScopedDiagnosticHandler {
public:
  JSONDiagnosticHandler(MLIRContext *ctx, llvm::raw_ostream &os)
      : ScopedDiagnosticHandler(ctx) {
    setHandler([&](Diagnostic &diag) {
      std::optional<FileLineColLoc> maybeFileLoc =
          extractFileLoc(diag.getLocation());

      if (!maybeFileLoc)
        return failure();

      FileLineColLoc fileLoc = *maybeFileLoc;

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
