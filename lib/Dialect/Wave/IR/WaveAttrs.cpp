// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "water/Dialect/Wave/IR/WaveEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "water/Dialect/Wave/IR/WaveAttrs.cpp.inc"

void wave::WaveDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "water/Dialect/Wave/IR/WaveAttrs.cpp.inc"
      >();
}

static llvm::LogicalResult
verifyNormalFormTypeRange(std::optional<mlir::Location> loc,
                          mlir::TypeRange types, wave::WaveNormalForm form,
                          llvm::StringRef message) {
  for (mlir::Type type : types) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(type);
    if (!tensorType || tensorType.getFullySpecified())
      continue;

    if (loc)
      mlir::emitError(*loc) << message;
    return llvm::failure();
  }
  return llvm::success();
}

llvm::LogicalResult wave::detail::verifyNormalFormAttr(
    mlir::Operation *root, wave::WaveNormalForm form, bool emitDiagnostics) {
  // No normal form required.
  if (static_cast<uint32_t>(form) == 0)
    return llvm::success();

  // Walk in pre-order so we hit functions sooner and verify them for the first
  // form.
  mlir::WalkResult walkResult =
      root->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
        std::optional<mlir::Location> optionalLoc;
        if (emitDiagnostics)
          optionalLoc = op->getLoc();

        if (auto func = llvm::dyn_cast<mlir::FunctionOpInterface>(op)) {
          if (wave::bitEnumContainsAll(
                  form, wave::WaveNormalForm::FunctionBoundarySpecified)) {
            const llvm::StringLiteral kMessage =
                "normal form requires tensor types to be fully specified at "
                "function boundaries";
            if (llvm::failed(verifyNormalFormTypeRange(
                    optionalLoc, func.getArgumentTypes(), form, kMessage)))
              return mlir::WalkResult::interrupt();
            if (llvm::failed(verifyNormalFormTypeRange(
                    optionalLoc, func->getResultTypes(), form, kMessage)))
              return mlir::WalkResult::interrupt();
          }
        }

        if (wave::bitEnumContainsAll(form,
                                     wave::WaveNormalForm::OpTypesSpecified)) {
          const llvm::StringLiteral kMessage =
              "normal form requires tensor types to be fully specified";
          if (llvm::failed(verifyNormalFormTypeRange(
                  optionalLoc, op->getOperandTypes(), form, kMessage)))
            return mlir::WalkResult::interrupt();
          if (llvm::failed(verifyNormalFormTypeRange(
                  optionalLoc, op->getResultTypes(), form, kMessage)))
            return mlir::WalkResult::interrupt();
          for (mlir::Region &region : op->getRegions()) {
            for (mlir::Block &block : region) {
              if (llvm::failed(verifyNormalFormTypeRange(
                      optionalLoc, block.getArgumentTypes(), form, kMessage)))
                return mlir::WalkResult::interrupt();
            }
          }
        }

        return mlir::WalkResult::advance();
      });

  return llvm::failure(walkResult.wasInterrupted());
}
