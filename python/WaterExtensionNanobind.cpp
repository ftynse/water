// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "water/c/Dialects.h"

#include "nanobind/nanobind.h"

#include "mlir/CAPI/Support.h"
#include "mlir/Support/TypeID.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"

namespace nb = nanobind;

static MlirTypeID getWaveSymbolAttrTypeID() {
  return wrap(mlir::TypeID::get<wave::WaveSymbolAttr>());
}

NB_MODULE(_waterDialects, m) {
  auto d = m.def_submodule("wave");
  d.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle h = mlirGetDialectHandle__wave__();
        mlirDialectHandleRegisterDialect(h, context);
        if (load)
          mlirDialectHandleLoadDialect(h, context);
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);

  //===---------------------------------------------------------------------===//
  // WaveSymbolAttr
  //===---------------------------------------------------------------------===//

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      d, "WaveSymbolAttr", mlirAttributeIsAWaveSymbolAttr,
      getWaveSymbolAttrTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::string &symbolName,
             // MlirContext should always come last to allow for being
             // automatically deduced from context.
             MlirContext context) {
            MlirStringRef symbolNameStrRef =
                mlirStringRefCreate(symbolName.data(), symbolName.size());
            return cls(mlirWaveSymbolAttrGet(context, symbolNameStrRef));
          },
          nb::arg("cls"), nb::arg("symbolName"),
          nb::arg("context") = nb::none(),
          "Gets a wave.wave_symbol from parameters.");
}
