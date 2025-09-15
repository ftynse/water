// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/CAPI/AffineMap.h"
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

//===---------------------------------------------------------------------===//
// WaveIndexMappingAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAWaveIndexMappingAttr(MlirAttribute attr) {
  return llvm::isa<wave::WaveIndexMappingAttr>(unwrap(attr));
}

MlirAttribute
mlirWaveIndexMappingAttrGet(MlirContext mlirCtx, MlirAttribute *symbolNames,
                            size_t numSymbolNames, MlirAffineMap start,
                            MlirAffineMap step, MlirAffineMap stride) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);

  // Convert C array of MlirAttribute to vector of WaveSymbolAttr
  llvm::SmallVector<wave::WaveSymbolAttr> symbolAttrs;
  symbolAttrs.reserve(numSymbolNames);
  for (size_t i = 0; i < numSymbolNames; ++i) {
    mlir::Attribute attr = unwrap(symbolNames[i]);
    if (auto symbolAttr = llvm::dyn_cast<wave::WaveSymbolAttr>(attr)) {
      symbolAttrs.push_back(symbolAttr);
    } else {
      return MlirAttribute{nullptr};
    }
  }

  return wrap(wave::WaveIndexMappingAttr::get(ctx, symbolAttrs, unwrap(start),
                                              unwrap(step), unwrap(stride)));
}

MlirTypeID mlirWaveIndexMappingAttrGetTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveIndexMappingAttr>());
}
