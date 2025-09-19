// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

#include "mlir/IR/Dialect.h"

#include "water/Dialect/Wave/IR/WaveDialect.cpp.inc"

void wave::WaveDialect::initialize() {
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "water/Dialect/Wave/IR/WaveOps.cpp.inc"
      >();
  registerTypes();
}

llvm::LogicalResult
wave::WaveDialect::verifyOperationAttribute(mlir::Operation *op,
                                            mlir::NamedAttribute attr) {
  if (attr.getName() == kNormalFormAttrName) {
    if (auto enumAttr = llvm::dyn_cast<WaveNormalFormAttr>(attr.getValue())) {
      return detail::verifyNormalFormAttr(op, enumAttr.getValue(),
                                          /*emitDiagnostics=*/true);
    }
    return op->emitError() << attr.getName() << " expects a WaveNormalFormAttr";
  }
  if (attr.getName() == kHyperparameterAttrName) {
    if (!llvm::isa<wave::WaveHyperparameterAttr>(attr.getValue())) {
      return op->emitError()
             << attr.getName() << " expects a WaveHyperparameterAttr";
    }

    // TODO: consider a mode where parameters can be be union'ed, but not
    // redefined. There are passes that currently assume a single set of
    // hyperparameters.
    for (mlir::Operation *parent = op->getParentOp(); parent != nullptr;
         parent = parent->getParentOp()) {
      if (parent->hasAttr(kHyperparameterAttrName)) {
        mlir::InFlightDiagnostic diag =
            op->emitError()
            << "defines hyperparameters when its ancestor already had";
        diag.attachNote(parent->getLoc()) << "ancestor";
        return diag;
      }
    }
    return llvm::success();
  }
  return op->emitError() << "unexpected wave dialect attribute "
                         << attr.getName() << " = " << attr.getValue();
}
