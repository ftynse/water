// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include "water/Dialect/Wave/IR/WaveEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "water/Dialect/Wave/IR/WaveAttrs.cpp.inc"

using namespace mlir;
using namespace wave;

//===----------------------------------------------------------------------===//
// WaveIndexMappingAttr
//===----------------------------------------------------------------------===//

Attribute WaveIndexMappingAttr::parse(AsmParser &parser, Type type) {
  // Parse custom syntax: '[' symbol-names ']' '->' '(' affine-expr ')'
  // This preserves meaningful symbol names while leveraging the existing
  // affine parser.

  SmallVector<Attribute> symbolNameAttrs;
  SmallVector<StringRef> symbolNames;

  // Parse '[' symbol-names ']' also allowing an empty list.
  if (parser.parseLSquare())
    return {};

  auto parseSymbol = [&]() -> ParseResult {
    StringRef symbolName;
    if (failed(parser.parseKeyword(&symbolName)))
      return failure();
    symbolNameAttrs.push_back(parser.getBuilder().getStringAttr(symbolName));
    symbolNames.push_back(symbolName);
    return success();
  };

  // If the list is empty, we should be able to immediately parse a ']'.
  // Otherwise, parse a non-empty, comma-separated list followed by ']'.
  if (failed(parser.parseOptionalRSquare())) {
    if (parser.parseCommaSeparatedList(parseSymbol) || parser.parseRSquare())
      return {};
  }

  // Parse '->' '(' affine-expr ')'
  if (parser.parseArrow() || parser.parseLParen())
    return {};

  auto context = parser.getContext();

  // Create symbol mapping for affine expression parser
  SmallVector<std::pair<StringRef, AffineExpr>> symbolSet;
  for (auto [i, symbolName] : llvm::enumerate(symbolNames)) {
    auto symbolExpr = getAffineSymbolExpr(i, context);
    symbolSet.emplace_back(symbolName, symbolExpr);
  }

  // Parse affine expression with symbol names
  AffineExpr parsedExpr;
  if (parser.parseAffineExpr(symbolSet, parsedExpr).failed() ||
      parser.parseRParen())
    return {};

  // Build the final attribute
  auto affineMap = AffineMap::get(
      /*numDims=*/0, /*numSymbols=*/symbolNames.size(), parsedExpr, context);
  auto mapAttr = AffineMapAttr::get(affineMap);
  ArrayAttr symbolNamesAttr = parser.getBuilder().getArrayAttr(symbolNameAttrs);

  return get(parser.getContext(), symbolNamesAttr, mapAttr);
}

void WaveIndexMappingAttr::print(AsmPrinter &printer) const {
  // Print custom syntax: '[' symbol-names ']' '->' '(' affine-expr ')' with
  // substituted symbol names
  printer << "[";
  llvm::interleaveComma(getSymbolNames().getAsRange<StringAttr>(), printer,
                        [&](StringAttr s) { printer << s.getValue(); });
  printer << "] -> ";

  // Get the affine expr (uses s0, s1, ... for symbols) and render it to a
  // string.
  AffineMap map = getAffineMap().getValue();
  AffineExpr expr = map.getResult(0);
  std::string exprStr;
  llvm::raw_string_ostream stream(exprStr);
  expr.print(stream);
  stream.flush();

  // Substitute symbol names
  auto symbolNames = getAllSymbolNames();
  for (auto [i, symbolName] : llvm::enumerate(symbolNames)) {
    std::string symbolPattern = "s" + std::to_string(i);

    // Use regex-like replacement to avoid partial matches
    size_t pos = 0;
    while ((pos = exprStr.find(symbolPattern, pos)) != std::string::npos) {
      // Check if this is a complete symbol (not part of a larger identifier)
      bool isCompleteSymbol =
          (pos == 0 || !std::isalnum(exprStr[pos - 1])) &&
          (pos + symbolPattern.length() == exprStr.length() ||
           !std::isalnum(exprStr[pos + symbolPattern.length()]));

      if (isCompleteSymbol) {
        exprStr.replace(pos, symbolPattern.length(), symbolName.str());
        pos += symbolName.size();
      } else {
        pos += symbolPattern.length();
      }
    }
  }

  printer << "(" << exprStr << ")";
}

void wave::WaveDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "water/Dialect/Wave/IR/WaveAttrs.cpp.inc"
      >();
}
