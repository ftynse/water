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
#include "water/Dialect/Wave/IR/WaveTypes.h"

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

// Return `true` if two tensor types have the same shape. Null types are
// considered to have different shapes.
static bool hasSameShape(wave::WaveTensorType lhs, wave::WaveTensorType rhs) {
  // TODO: this may require more advanced checking if shapes are more complex
  // than a single symbol.
  return lhs && rhs && lhs.getShape() == rhs.getShape();
}

llvm::FailureOr<mlir::ChangeResult> wave::detail::checkPropagateShapeConflict(
    wave::WaveTensorType from, wave::WaveTensorType to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  if (!from || !to || hasSameShape(from, to))
    return mlir::ChangeResult::NoChange;

  if (!to.getFullySpecified())
    return mlir::ChangeResult::Change;

  errs << "irreconcilable types during type inference from " << fromName << "("
       << from << ") to " << toName << "(" << to << ")";
  return mlir::failure();
}

llvm::FailureOr<mlir::ChangeResult> wave::detail::propagateShapeInformation(
    wave::WaveTensorType from, wave::WaveTensorType &to,
    llvm::StringRef fromName, llvm::StringRef toName, llvm::raw_ostream &errs) {
  llvm::FailureOr<mlir::ChangeResult> res =
      checkPropagateShapeConflict(from, to, fromName, toName, errs);
  if (mlir::failed(res) || *res == mlir::ChangeResult::NoChange)
    return res;

  to = to.copyShapeFrom(from);
  return mlir::ChangeResult::Change;
}

llvm::FailureOr<mlir::ChangeResult>
wave::detail::identityTypeInferencePropagate(
    llvm::ArrayRef<wave::WaveTensorType> from,
    llvm::MutableArrayRef<wave::WaveTensorType> to, llvm::StringRef fromName,
    llvm::StringRef toName, llvm::raw_ostream &errs) {
  auto it = llvm::find_if(from, [](wave::WaveTensorType type) {
    return type && type.getFullySpecified();
  });
  if (it == from.end())
    return mlir::ChangeResult::NoChange;

  // Expect all fully-specified "from" types to have the same shape.
  for (auto [i, fr] : llvm::enumerate(from)) {
    llvm::FailureOr<mlir::ChangeResult> res =
        checkPropagateShapeConflict(*it, fr, fromName, toName, errs);
    if (mlir::failed(res)) {
      errs << " for " << fromName << " #" << i;
      return res;
    }
  }

  mlir::ChangeResult changeResult = mlir::ChangeResult::NoChange;
  for (auto &&[i, toType] : llvm::enumerate(to)) {
    llvm::FailureOr<mlir::ChangeResult> res =
        propagateShapeInformation(*it, toType, fromName, toName, errs);
    if (mlir::failed(res)) {
      errs << " for " << fromName << " #" << i;
      return mlir::failure();
    }

    changeResult |= *res;
  }
  return changeResult;
}

#include "water/Dialect/Wave/IR/WaveOpInterfaces.cpp.inc"
