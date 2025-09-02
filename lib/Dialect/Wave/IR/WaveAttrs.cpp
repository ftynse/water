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
  // Parse custom syntax: '[' symbol-names ']' '->' '(' start, step, stride ')'
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

  // Parse affine expr triple: '->' '(' start_expr ',' step_expr ',' stride_expr
  // ')'
  if (parser.parseArrow() || parser.parseLParen())
    return {};

  MLIRContext *context = parser.getContext();
  auto parseExprWithNames = [&](ArrayRef<StringRef> names,
                                AffineExpr &outExpr) -> ParseResult {
    SmallVector<std::pair<StringRef, AffineExpr>> symbolSet;
    symbolSet.reserve(names.size());
    for (auto [i, nm] : llvm::enumerate(names))
      symbolSet.emplace_back(nm, getAffineSymbolExpr(i, context));
    if (failed(parser.parseAffineExpr(symbolSet, outExpr)))
      return failure();
    return success();
  };

  AffineExpr startExpr;
  AffineExpr stepExpr;
  AffineExpr strideExpr;
  if (failed(parseExprWithNames(symbolNames, startExpr)))
    return {};
  if (parser.parseComma())
    return {};
  if (failed(parseExprWithNames(symbolNames, stepExpr)))
    return {};
  if (parser.parseComma())
    return {};
  if (failed(parseExprWithNames(symbolNames, strideExpr)))
    return {};

  if (parser.parseRParen())
    return {};

  // Build maps
  auto startMap = AffineMap::get(
      /*numDims=*/0, /*numSymbols=*/symbolNames.size(), startExpr, context);
  auto stepMap = AffineMap::get(
      /*numDims=*/0, /*numSymbols=*/symbolNames.size(), stepExpr, context);
  auto strideMap = AffineMap::get(
      /*numDims=*/0, /*numSymbols=*/symbolNames.size(), strideExpr, context);

  ArrayAttr symbolNamesAttr = parser.getBuilder().getArrayAttr(symbolNameAttrs);
  return get(parser.getContext(), symbolNamesAttr, AffineMapAttr::get(startMap),
             AffineMapAttr::get(stepMap), AffineMapAttr::get(strideMap));
}

void WaveIndexMappingAttr::print(AsmPrinter &printer) const {
  // Print '[' symbol-names '] -> (start, step, stride)'.
  // We keep one global symbol list (symbol_names) for all three expressions.
  // Each expression is an affine map with the same numSymbols; we substitute
  // s0, s1, ... using the shared names when rendering each expression.
  printer << "[";
  llvm::interleaveComma(getSymbolNames().getAsRange<StringAttr>(), printer,
                        [&](StringAttr s) { printer << s.getValue(); });
  printer << "] -> ";

  // Helper: render an affine map result to a string, then textual-substitute
  // s<i> occurrences with the corresponding symbol_names[i].
  auto stringifyWithNames = [&](AffineMap map,
                                ArrayRef<StringRef> names) -> std::string {
    AffineExpr expr = map.getResult(0);
    std::string exprStr;
    llvm::raw_string_ostream os(exprStr);
    expr.print(os);
    os.flush();
    for (auto [i, nm] : llvm::enumerate(names)) {
      std::string pattern = "s" + std::to_string(i);
      size_t pos = 0;
      while ((pos = exprStr.find(pattern, pos)) != std::string::npos) {
        // Replace only when 'pattern' is a complete token (not embedded
        // inside a longer identifier or number). We approximate token
        // boundaries by checking that adjacent characters are non-alphanumeric.
        bool isWhole = (pos == 0 || !std::isalnum(exprStr[pos - 1])) &&
                       (pos + pattern.length() == exprStr.length() ||
                        !std::isalnum(exprStr[pos + pattern.length()]));
        if (isWhole) {
          exprStr.replace(pos, pattern.length(), nm.str());
          pos += nm.size();
        } else {
          pos += pattern.length();
        }
      }
    }
    return exprStr;
  };

  auto allNames = getAllSymbolNames();
  // All three maps share the same symbol set and order.
  auto startStr = stringifyWithNames(getStart().getValue(), allNames);
  auto stepStr = stringifyWithNames(getStep().getValue(), allNames);
  auto strideStr = stringifyWithNames(getStride().getValue(), allNames);

  printer << "(" << startStr << ", " << stepStr << ", " << strideStr << ")";
}

void wave::WaveDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "water/Dialect/Wave/IR/WaveAttrs.cpp.inc"
      >();
}
