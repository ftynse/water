// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "water/Dialect/Wave/IR/WaveAttrs.td"
include "water/Dialect/Wave/IR/WaveDialect.td"
include "water/Dialect/Wave/IR/WaveTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

#ifndef WATER_DIALECT_WAVE_WAVEOPS
#define WATER_DIALECT_WAVE_WAVEOPS

//-----------------------------------------------------------------------------
// Base classes for arithmetic operations
//-----------------------------------------------------------------------------

class UnaryWaveOp<string mnemonic> : Op<WaveDialect, mnemonic> {
  let arguments = (ins
    Arg<WaveTensorType, "Argument">:$argument
  );

  let results = (outs
    Arg<WaveTensorType, "Result">:$result
  );

  let assemblyFormat = "$argument attr-dict `:` functional-type(operands, results)";
}

class BinaryWaveOp<string mnemonic> : Op<WaveDialect, mnemonic> {
  let arguments = (ins
    Arg<WaveTensorType, "Left-hand side">:$lhs,
    Arg<WaveTensorType, "Right-hand side">:$rhs
  );

  let results = (outs
    Res<WaveTensorType, "Result of the operation">:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

//-----------------------------------------------------------------------------
// Arithmetic operations
//-----------------------------------------------------------------------------

def AddOp : BinaryWaveOp<"add"> {
  let summary = "Add two values";
}

def MulOp : BinaryWaveOp<"mul"> {
  let summary = "Multiply two values";
}

def DivOp : BinaryWaveOp<"div"> {
  let summary = "Divide two values";
}

def Exp2Op : UnaryWaveOp<"exp2"> {
  let summary = "Compute 2 to the power of the argument";
}

def MmaOp : Op<WaveDialect, "mma"> {
  let summary = "Matrix multiply and accumulate";

  let arguments = (ins
    Arg<WaveTensorType, "Left-hand side of the multiplication">:$lhs,
    Arg<WaveTensorType, "Right-hand side of the multiplication">:$rhs,
    Arg<WaveTensorType, "Accumulator for addition">:$accumulator,
    Arg<WaveMmaKindAttr, "Kind of the MMA intrinsic to target">:$kind
  );

  let results = (outs
    Res<WaveTensorType, "Accumulated result">:$result
  );

  let assemblyFormat = "$lhs `,` $rhs `,` $accumulator attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

//-----------------------------------------------------------------------------
// Control flow operations
//-----------------------------------------------------------------------------

def IterateOp : Op<WaveDialect, "iterate", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface, ["areTypesCompatible",
                                                        "getEntrySuccessorOperands"]>]> {
  let summary = "Executes the body repeatedly";

  let arguments = (ins
    Arg<WaveSymbolAttr, "Iterator symbol">:$iterator,
    Arg<Variadic<WaveTensorType>, "Carried values">:$iter_args
  );

  let results = (outs
    Res<Variadic<WaveTensorType>, "Yielded values">:$results
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = "custom<SingleSymbol>($iterator) (`iter_args` `(` $iter_args^ `)`)? attr-dict-with-keyword regions `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def YieldOp : Op<WaveDialect, "yield",
    [Terminator, HasParent<"::wave::IterateOp">,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
  let summary = "Yields values from the current control flow context";

  let arguments = (ins
    Arg<Variadic<WaveTensorType>, "Yielded values">:$values
  );

  let assemblyFormat = "$values attr-dict `:` type($values)";
}

//-----------------------------------------------------------------------------
// Memory-related operations
//-----------------------------------------------------------------------------

def ReadOp : Op<WaveDialect, "read"> {
  let summary = "Reads from memory";

  let arguments = (ins
    Arg<WaveTensorInMemory, "Tensor representing memory to read from">:$memory
  );

  let results = (outs
    Res<WaveTensorInRegister, "Read value">:$result
  );

  let assemblyFormat = "$memory attr-dict `:` functional-type(operands, results)";
}

def RegisterOp : Op<WaveDialect, "register"> {
  let summary = "Defines a tensor value known to be placed in a register";

  let arguments = (ins
    Arg<Type<Or<[AnyInteger.predicate, AnyFloat.predicate]>>, "Scalar value to initialize the tensor elements">:$init
  );

  let results = (outs
    Res<WaveTensorInRegister, "Defined value">:$result
  );

  let assemblyFormat = "$init attr-dict `:` custom<RegisterOpTypes>(type($init), type($result))";
  let hasVerifier = 1;
}

def WriteOp : Op<WaveDialect, "write"> {
  let summary = "Writes into memory";

  let arguments = (ins
    Arg<WaveTensorInRegister, "Value to write">:$value_to_store,
    Arg<WaveTensorInMemory, "Tensor representing memory to write into">:$memory
  );

  let assemblyFormat = "$value_to_store `,` $memory attr-dict `:` type($value_to_store) `,` type($memory)";
}

#endif // WATER_DIALECT_WAVE_WAVEOPS
