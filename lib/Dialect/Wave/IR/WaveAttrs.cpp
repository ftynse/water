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

  SmallVector<WaveSymbolAttr> symbolNameAttrs;
  SmallVector<StringRef> symbolNames;

  auto parseSymbol = [&]() -> ParseResult {
    StringRef symbolName;
    if (failed(parser.parseKeyword(&symbolName)))
      return failure();
    symbolNameAttrs.push_back(
        WaveSymbolAttr::get(parser.getContext(), symbolName));
    symbolNames.push_back(symbolName);
    return success();
  };

  // Parse '[' symbol-names ']' allowing empty or non-empty lists.
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseSymbol))
    return {};

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
    return parser.parseAffineExpr(symbolSet, outExpr);
  };

  AffineExpr startExpr;
  AffineExpr stepExpr;
  AffineExpr strideExpr;
  if (failed(parseExprWithNames(symbolNames, startExpr)) ||
      parser.parseComma() ||
      failed(parseExprWithNames(symbolNames, stepExpr)) ||
      parser.parseComma() ||
      failed(parseExprWithNames(symbolNames, strideExpr)) ||
      parser.parseRParen()) {
    parser.emitError(
        parser.getCurrentLocation(),
        "expected three affine expressions for '(start, step, stride)'");
    return {};
  }

  // Build maps
  auto startMap = AffineMap::get(
      /*numDims=*/0, /*numSymbols=*/symbolNames.size(), startExpr, context);
  auto stepMap = AffineMap::get(
      /*numDims=*/0, /*numSymbols=*/symbolNames.size(), stepExpr, context);
  auto strideMap = AffineMap::get(
      /*numDims=*/0, /*numSymbols=*/symbolNames.size(), strideExpr, context);

  return get(parser.getContext(), symbolNameAttrs, startMap, stepMap,
             strideMap);
}

void WaveIndexMappingAttr::print(AsmPrinter &printer) const {
  // Print '[' symbol-names '] -> (start, step, stride)'.
  // We keep one global symbol list (symbol_names) for all three expressions.
  // Each expression is an affine map with the same numSymbols; we substitute
  // s0, s1, ... using the shared names when rendering each expression.
  printer << "[";
  llvm::interleaveComma(getSymbolNames(), printer,
                        [&](WaveSymbolAttr s) { printer << s.getName(); });
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

  SmallVector<StringRef> allNames = getAllSymbolNames();
  // All three maps share the same symbol set and order.
  std::string startStr = stringifyWithNames(getStart(), allNames);
  std::string stepStr = stringifyWithNames(getStep(), allNames);
  std::string strideStr = stringifyWithNames(getStride(), allNames);

  printer << "(" << startStr << ", " << stepStr << ", " << strideStr << ")";
}

//===----------------------------------------------------------------------===//
// WaveHyperparameterAttr
//===----------------------------------------------------------------------===//

std::optional<int64_t>
WaveHyperparameterAttr::getSymbolValue(StringRef symbolName) const {
  auto mapping = getMapping();
  auto attr = mapping.get(symbolName);
  if (!attr)
    return std::nullopt;

  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();

  return std::nullopt;
}

bool WaveHyperparameterAttr::hasSymbol(StringRef symbolName) const {
  return getMapping().get(symbolName) != nullptr;
}

void wave::WaveDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "water/Dialect/Wave/IR/WaveAttrs.cpp.inc"
      >();
}
