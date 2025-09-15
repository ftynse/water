// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/c/Dialects.h"
#include "mlir/CAPI/Registration.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"

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
