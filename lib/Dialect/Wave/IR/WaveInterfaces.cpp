// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"

using namespace mlir;

LogicalResult wave::verifyWaveIndexMappings(Operation *op) {
  auto dict = op->getAttrDictionary();
  auto indexNamed = dict.getNamed("index");
  if (!indexNamed)
    return success();

  auto dictAttr = dyn_cast<DictionaryAttr>(indexNamed->getValue());
  if (!dictAttr)
    return op->emitError("'index' attribute must be a dictionary");

  for (auto named : dictAttr) {
    auto val = named.getValue();
    if (!isa<wave::WaveIndexMappingAttr>(val))
      return op->emitError(
                 "index attribute values must be WaveIndexMappingAttr, got ")
             << val;

    auto mapping = cast<wave::WaveIndexMappingAttr>(val);
    auto checkNoDims = [&](AffineMap map, StringRef which) -> LogicalResult {
      if (map.getNumDims() != 0)
        return op->emitError(
                   "wave indexing " + which +
                   " map should have no dimensions, only symbols, got ")
               << map.getNumDims() << " dimensions for symbol "
               << named.getName();
      return success();
    };

    AffineMap startMap = mapping.getStart();
    AffineMap stepMap = mapping.getStep();
    AffineMap strideMap = mapping.getStride();
    if (failed(checkNoDims(startMap, "start")) ||
        failed(checkNoDims(stepMap, "step")) ||
        failed(checkNoDims(strideMap, "stride")))
      return failure();

    unsigned declared = mapping.getSymbolNames().size();
    if (startMap.getNumSymbols() != declared ||
        stepMap.getNumSymbols() != declared ||
        strideMap.getNumSymbols() != declared)
      return op->emitError("inconsistent symbol count between symbol_names and "
                           "affine maps for index symbol '")
             << named.getName() << "' (expected " << declared
             << ", got start=" << startMap.getNumSymbols()
             << ", step=" << stepMap.getNumSymbols()
             << ", stride=" << strideMap.getNumSymbols() << ")";
  }
  return success();
}

// ODS custom directive: parseWaveIndexDict/printWaveIndexDict
ParseResult wave::parseWaveIndexDict(OpAsmParser &parser, DictionaryAttr &out) {
  llvm::SmallVector<NamedAttribute, 4> mappingDict;
  if (parser.parseLBrace())
    return failure();

  auto parseOne = [&]() -> ParseResult {
    llvm::StringRef symbolName;
    if (parser.parseKeyword(&symbolName) || parser.parseColon())
      return failure();

    // Parse WaveIndexMappingAttr using its custom parser to allow concise
    // syntax.
    auto indexMapping = wave::WaveIndexMappingAttr::parse(parser, Type{});
    if (!indexMapping)
      return failure();

    mappingDict.emplace_back(parser.getBuilder().getStringAttr(symbolName),
                             indexMapping);
    return success();
  };

  if (parser.parseCommaSeparatedList(parseOne) || parser.parseRBrace())
    return failure();

  out = parser.getBuilder().getDictionaryAttr(mappingDict);
  return success();
}

void wave::printWaveIndexDict(OpAsmPrinter &printer, Operation *op,
                              DictionaryAttr dict) {
  printer.getStream() << "{";
  llvm::interleaveComma(
      dict, printer.getStream(), [&](NamedAttribute namedAttr) {
        printer.getStream() << namedAttr.getName().getValue() << " : ";
        if (auto mappingAttr = llvm::dyn_cast<wave::WaveIndexMappingAttr>(
                namedAttr.getValue())) {
          mappingAttr.print(printer);
        } else {
          printer.printAttribute(namedAttr.getValue());
        }
      });
  printer.getStream() << "}";
}
