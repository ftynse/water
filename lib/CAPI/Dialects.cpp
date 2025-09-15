// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/CAPI/Registration.h"
#include "mlir/Support/TypeID.h"

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/c/Dialects.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Wave, wave, ::wave::WaveDialect)

//===---------------------------------------------------------------------===//
// WaveSymbolAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveSymbolAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveSymbolAttr>(unwrap(attr));
}

MlirAttribute mlirWaveSymbolAttrGet(MlirContext mlirCtx,
                                    MlirStringRef symbolNameStrRef) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  llvm::StringRef symbolName = unwrap(symbolNameStrRef);
  return wrap(wave::WaveSymbolAttr::get(ctx, symbolName));
}

MlirTypeID mlirWaveSymbolAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveSymbolAttr>());
}
