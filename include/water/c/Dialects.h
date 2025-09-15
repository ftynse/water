// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_C_DIALECTS_H
#define WATER_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Wave, wave);

//===---------------------------------------------------------------------===//
// WaveSymbolAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveSymbolAttr.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAWaveSymbolAttr(MlirAttribute attr);

/// Creates a new WaveSymbolAttr with the given symbol name.
MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveSymbolAttrGet(MlirContext mlirCtx, MlirStringRef symbolName);

/// Returns the typeID of a WaveSymbolAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveSymbolAttrGetTypeID();

//===---------------------------------------------------------------------===//
// WaveIndexMappingAttr
//===---------------------------------------------------------------------===//

/// Checks whether the given MLIR attribute is a WaveIndexMappingAttr.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsAWaveIndexMappingAttr(MlirAttribute attr);

/// Creates a new WaveIndexMappingAttr with the given symbol name.
MLIR_CAPI_EXPORTED MlirAttribute mlirWaveIndexMappingAttrGet(
    MlirContext mlirCtx, MlirAttribute *symbolNames, size_t numSymbolNames,
    MlirAffineMap start, MlirAffineMap step, MlirAffineMap stride);

/// Returns the typeID of a WaveIndexMappingAttr.
MLIR_CAPI_EXPORTED MlirTypeID mlirWaveIndexMappingAttrGetTypeID();

#ifdef __cplusplus
}
#endif

#endif // WATER_C_DIALECTS_H
