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

MLIR_CAPI_EXPORTED bool mlirAttributeIsAWaveSymbolAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirWaveSymbolAttrGet(MlirContext mlirCtx, MlirStringRef symbolName);

#ifdef __cplusplus
}
#endif

#endif // WATER_C_DIALECTS_H
