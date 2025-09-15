// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "water/c/Dialects.h"

namespace nb = nanobind;

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

  m.attr("WaveSymbolAttr") = mlir_attribute_subclass(m, "WaveSymbolAttr", mlirAttributeIsAWaveSymbolAttr)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirContext context, const nb::bytes &symbolName) {
                      std::cout << "WaveSymbolAttr.get called" << std::endl;

            MlirStringRef symbolNameStrRef = mlirStringRefCreate(
                static_cast<char *>(const_cast<void *>(symbolName.data())),
                symbolName.size());
            return cls(mlirWaveSymbolAttrGet(context, symbolNameStrRef));
          },
          "cls"_a, "context"_a, "symbol"_a,
          "Gets a wave.wave_symbol from parameters.")
}
