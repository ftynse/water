// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "water/Dialect/Wave/IR/WaveDialect.td"
include "water/Dialect/Wave/IR/WaveTypes.td"
include "water/Dialect/Wave/IR/WaveAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

#ifndef WATER_DIALECT_WAVE_WAVEOPS
#define WATER_DIALECT_WAVE_WAVEOPS

def MmaOp : Op<WaveDialect, "mma"> {
  let summary = "Matrix multiply and accumulate";

  let arguments = (ins
    Arg<WaveTensorType, "Left-hand side of the multiplication">:$lhs,
    Arg<WaveTensorType, "Right-hand side of the multiplication">:$rhs,
    Arg<WaveTensorType, "Accumulator for addition">:$accumulator
  );

  let results = (outs
    Res<WaveTensorType, "Accumulated result">:$result
  );

  let assemblyFormat = "$lhs `,` $rhs `,` $accumulator attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}


def RegisterOp : Op<WaveDialect, "register"> {
  let summary = "Create a new register with specified shape, data type, and initial value";
  let description = [{
    The `wave.register` operation creates a new register storage location with
    symbolic shape dimensions, element data type, and an initial value. This
    operation is used to allocate register storage that can be used in subsequent
    Wave dialect operations.

    The initial value is specified as a typed attribute and must be compatible with
    the element type of the result register.

    The optional `index` attribute defines indexing expressions for each dimension
    of the register using meaningful symbol names and affine expressions. Each
    dimension symbol is mapped to an affine expression with named symbols.

    Examples:
    ```mlir
    %reg = wave.register(0.0) : !wave.register<[@M, @N] of f32>

    %reg2 = wave.register(42.0) index {
      M : [WG0, BLOCK_M] -> (WG0 * BLOCK_M),
      N : [T1, BLOCK_N] -> (T1 * BLOCK_N + 8)
    } : !wave.register<[@M, @N] of f32>
    ```
  }];

  let arguments = (ins
    AnyAttr:$value,
    OptionalAttr<DictionaryAttr>:$index
  );

  let results = (outs
    Res<WaveRegisterType, "The created register">:$result
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // WATER_DIALECT_WAVE_WAVEOPS
