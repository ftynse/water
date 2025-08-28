// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WATER_DIALECT_WAVE_WAVEATTRS
#define WATER_DIALECT_WAVE_WAVEATTRS

include "water/Dialect/Wave/IR/WaveDialect.td"
include "mlir/IR/AttrTypeBase.td"

def WaveSymbolAttr : AttrDef<WaveDialect, "WaveSymbol"> {
  let mnemonic = "symbol";
  let description = [{
    Base symbol used in symbolic expressions in the Wave dialect. This is
    different from the SymbolRef hierarchy of attributes since it refers to
    symbols, aka variables, in mathematical expressions as opposed to symbols
    identifying global objects such as functions and global variables in
    programs. Wave expression symbols are not part of the symbol table.
  }];

  let parameters = (ins StringRefParameter<"name of the symbol">:$name);
  let assemblyFormat = "`<` $name `>`";

  let extraClassDeclaration = [{
    void printAsSymbolRef(::llvm::raw_ostream &os) const {
      os << "@" << getName();
    }
  }];
}

def WaveIndexMappingAttr : AttrDef<WaveDialect, "WaveIndexMapping"> {
  let mnemonic = "index_mapping";
  let description = [{
    An affine map with named symbols for Wave indexing expressions.

    This attribute preserves meaningful symbol names (e.g., WG0, BLOCK_M, T0)
    while storing an affine map internally. The symbol_names array
    corresponds 1:1 to the symbols in the affine map, where s0 maps to
    symbol_names[0], s1 to symbol_names[1], etc.

    Custom syntax: [symbol_names] -> (affine_expression)
    Example: [WG0, BLOCK_M] -> (WG0 * BLOCK_M + 42)
  }];

  let parameters = (ins
    "::mlir::ArrayAttr":$symbol_names,
    "::mlir::AffineMapAttr":$affine_map
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the symbol name at the given index.
    ::llvm::StringRef getSymbolName(unsigned index) {
      auto names = getSymbolNames().getValue();
      if (index >= names.size()) return ::llvm::StringRef();
      return ::mlir::cast<::mlir::StringAttr>(names[index]).getValue();
    }

    /// Get all symbol names as a vector of string references.
    ::llvm::SmallVector<::llvm::StringRef> getAllSymbolNames() const {
      ::llvm::SmallVector<::llvm::StringRef> result;
      for (auto attr : getSymbolNames().getValue()) {
        result.push_back(::mlir::cast<::mlir::StringAttr>(attr).getValue());
      }
      return result;
    }

    /// Get the underlying affine map.
    ::mlir::AffineMap getMap() { return getAffineMap().getValue(); }
  }];
}

#endif // WATER_DIALECT_WAVE_WAVEATTRS
