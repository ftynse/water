// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "water/Dialect/Wave/IR/WaveAttrs.td"
include "water/Dialect/Wave/IR/WaveDialect.td"
include "water/Dialect/Wave/IR/WaveInterfaces.td"
include "water/Dialect/Wave/IR/WaveTypes.td"
include "mlir/IR/OpBase.td"
include "water/Dialect/Wave/IR/WaveInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


#ifndef WATER_DIALECT_WAVE_WAVEOPS
#define WATER_DIALECT_WAVE_WAVEOPS

//-----------------------------------------------------------------------------
// Base class for all Wave operations.
//-----------------------------------------------------------------------------

// Wave operations must have an optional attribute for index sequences.
class WaveOp<string mnemonic, list<Trait> traits = []>
    : Op<WaveDialect, mnemonic, !listconcat(traits, [HasWaveIndexMapping])> {
  dag commonArguments = (ins
    Arg<OptionalAttr<DictionaryAttr>, "Index expression">:$index
  );

  string commonArgumentsSyntax = "( `index` custom<WaveIndexDict>($index)^ )?";
}

//-----------------------------------------------------------------------------
// Base classes for arithmetic operations
//-----------------------------------------------------------------------------

class WaveArithmeticOpDoc {
  string baseDescription = [{
    Integer overflow and signedness as well as floating point fastmath semantics
    are currently **UNDEFINED**.

    All operands and results are expected to live in registers. Previous
    operations must bring them to registers if needed.
  }];
}

class UnaryWaveOp<string mnemonic>
    : WaveOp<mnemonic,
         [WaveInferTypeOpInterface, IdentityTypeInferenceOpTrait]>,
      WaveArithmeticOpDoc {
  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Argument">:$argument
  ), commonArguments);

  let results = (outs
    Arg<WaveTensorInRegister, "Result">:$result
  );

  let assemblyFormat =
    "$argument attr-dict `:` functional-type(operands, results)";
}

class BinaryWaveOp<string mnemonic>
    : WaveOp<mnemonic,
         [WaveInferTypeOpInterface, IdentityTypeInferenceOpTrait]>,
      WaveArithmeticOpDoc {
  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Left-hand side">:$lhs,
    Arg<WaveTensorInRegister, "Right-hand side">:$rhs
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Result of the operation">:$result
  );

  let assemblyFormat =
    "$lhs `,` $rhs " # commonArgumentsSyntax # "attr-dict `:`"
    "functional-type(operands, results)";
}

//-----------------------------------------------------------------------------
// Arithmetic operations
//-----------------------------------------------------------------------------

def AddOp : BinaryWaveOp<"add"> {
  let summary = "Add two values";
  let description = baseDescription;
}

def MulOp : BinaryWaveOp<"mul"> {
  let summary = "Multiply two values";
  let description = baseDescription;
}

def DivOp : BinaryWaveOp<"div"> {
  let summary = "Divide two values";
  let description = baseDescription;
}

def Exp2Op : UnaryWaveOp<"exp2"> {
  let summary = "Compute 2 to the power of the argument";
  let description = baseDescription;
}

def MmaOp : WaveOp<"mma",
    [DeclareOpInterfaceMethods<WaveInferTypeOpInterface>]>,
            WaveArithmeticOpDoc {
  let summary = "Matrix multiply and accumulate";
  let description = baseDescription # [{
    The mandatory `kind` attribute indicates which configuration to use in
    emitting matrix core instructions. Operand and result tensors are expected
    to have the corresponding elemental types. In particular, the first
    component of the `kind`-s textual format indicates the accumulator and
    result elemental type and the last component indicates the LHS/RHS elemental
    type.
  }];

  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Left-hand side of the multiplication">:$lhs,
    Arg<WaveTensorInRegister, "Right-hand side of the multiplication">:$rhs,
    Arg<WaveTensorInRegister, "Accumulator for addition">:$accumulator,
    Arg<WaveMmaKindAttr, "Kind of the MMA intrinsic to target">:$kind
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Accumulated result">:$result
  );

  let assemblyFormat =
    "$lhs `,` $rhs `,` $accumulator " # commonArgumentsSyntax # "attr-dict `:`"
    "functional-type(operands, results)";
  let hasVerifier = 1;
}

//-----------------------------------------------------------------------------
// Control flow operations
//-----------------------------------------------------------------------------

def IterateOp : Op<WaveDialect, "iterate", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["areTypesCompatible", "getEntrySuccessorOperands"]>]> {
  let summary = "Executes the body repeatedly";
  let description = [{
    Intrinsically sequential iterative execution that is akin to a loop with
    bounds that are not yet specified. Instead, the iteration is understood to
    be performed along the symbolic dimension that will later be instantiated to
    a concrete value.

    Similarly to other loop-like constructs, this operation uses secondary
    induction variables for loop-carried values. The initial values are supplied
    as `iter_args` and the resulting values are produced by `result`.
  }];

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

  let assemblyFormat =
    "custom<SingleSymbol>($iterator) (`iter_args` `(` $iter_args^ `)`)?"
    "attr-dict-with-keyword regions `:` functional-type(operands, results)";
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

def ReadOp : WaveOp<"read", [
    WaveInferTypeOpInterface, IdentityTypeInferenceOpTrait]> {
  let summary = "Reads from memory";
  let description = [{
    Moves data from a memory-resident tensor to a register-resident tensor
    preserving the shape.
  }];

  let arguments = !con((ins
    Arg<WaveTensorInMemory, "Tensor representing memory to read from">:$memory
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Read value">:$result
  );

  let assemblyFormat = "$memory " # commonArgumentsSyntax # "attr-dict `:` "
                       "functional-type(operands, results)";
}

def RegisterOp : WaveOp<"register", [
    WaveInferTypeOpInterface, NoOpTypeInferenceOpTrait]> {
  let summary = "Defines a tensor value known to be placed in a register";
  let description = [{
    Defines a new register-resident tensor initialized with the given scalar
    value. The (symbolic) shape of the tensor may be specified in the result
    type.
  }];

  let arguments = !con((ins
    Arg<Type<Or<[AnyInteger.predicate, AnyFloat.predicate]>>,
        "Scalar value to initialize the tensor elements">:$init
  ), commonArguments);

  let results = (outs
    Res<WaveTensorInRegister, "Defined value">:$result
  );

  let assemblyFormat =
    "$init " # commonArgumentsSyntax # " attr-dict `:`"
    " custom<RegisterOpTypes>(type($init), type($result))";
  let hasVerifier = 1;
}

def WriteOp : WaveOp<"write", [
    WaveInferTypeOpInterface, NoOpTypeInferenceOpTrait]> {
  let summary = "Writes into memory";
  let description = [{
    Moves data from a register-resident tensor into a memory-resident tensor
    preserving the shape.
  }];

  let arguments = !con((ins
    Arg<WaveTensorInRegister, "Value to write">:$value_to_store,
    Arg<WaveTensorInMemory, "Tensor representing memory to write into">:$memory
  ), commonArguments);

  let assemblyFormat =
    "$value_to_store `,` $memory " # commonArgumentsSyntax # "attr-dict `:`"
    "type($value_to_store) `,` type($memory)";
}

def AllocateOp : WaveOp<"allocate"> {
  let summary = "Represents an allocation in an address space";
  let description = [{
    Allocates buffers with logical shape/type taken from the result
    type and with a separate distributed shape controlling the physical layout.
  }];
  // TODO: add padding/tail_padding arguments and and reflect them in distributed_shape computation
  let arguments = !con((ins
    Arg<ArrayAttr, "Size of the distributed shape">:$distributed_shape),
   commonArguments);

  let results = (outs Res<WaveTensorType,"Allocated buffer">:$result
  );

  let assemblyFormat = "`distributed_shape` `=` custom<SymbolList>($distributed_shape) attr-dict `:` type($result)";

  let hasVerifier = 1;
}


def AddrSpaceCastOp : WaveOp<"addrspace_cast",[DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Casts a handle with type wavetensortype to another address space";
  let description = [{
    Produces a new SSA handle to the same logical tensor but with a different
    address space attribute. The shape and element type must be identical
    between source and result, only the address space may change.

    This is used by the promotion passes for instance to obtain a <global>-typed handle
    for seeding a newly allocated <shared> buffer (global → register → LDS).
  }];

  let arguments = !con((ins
    Arg<WaveTensorType, "Source handle to cast">:$source,
    Arg<WaveAddressSpaceAttr,"Address space to target">:$target), commonArguments);

  let results = (outs
    Res<WaveTensorType, "Result handle in the target address space">:$results
  );

  let assemblyFormat = "$source  " # commonArgumentsSyntax # "attr-dict `:`  functional-type(operands, results)";

  let hasVerifier = 1;
}

#endif // WATER_DIALECT_WAVE_WAVEOPS
