// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace wave;

#define GET_OP_CLASSES
#include "water/Dialect/Wave/IR/WaveOps.cpp.inc"

// Update negative indices in the array to positive equivalents given the total
// rank, e.g. -1 and -3 get updated to 3 and 1, respectively, for the rank of 4.
static void updateNegativeIndices(llvm::MutableArrayRef<int> indices,
                                  int rank) {
  for (int &index : indices) {
    if (index < 0)
      index += rank;
  }
}

// Verify that specified dimensions match between LHS and RHS, the lists of
// dimensions are expected to be co-indexed. Emit diagnostic errors and return
// failure when it is not the case.
static LogicalResult
verifyTypesMatchingDimensions(Location loc, llvm::StringRef lhsName,
                              WaveTensorType lhs, llvm::ArrayRef<int> lhsDims,
                              llvm::StringRef rhsName, WaveTensorType rhs,
                              llvm::ArrayRef<int> rhsDims) {
  // TODO: check whether it is possible to turn this into a trait.

  assert(lhsDims.size() == rhsDims.size() &&
         "expected lhs and rhs dim lists to be co-indexed");

  // Under-specified types are okay everywhere.
  if (!lhs.getFullySpecified() || !rhs.getFullySpecified())
    return success();

  llvm::SmallVector<int> lhsDimsVec(lhsDims), rhsDimsVec(rhsDims);
  updateNegativeIndices(lhsDimsVec, lhs.getRank());
  updateNegativeIndices(rhsDimsVec, rhs.getRank());
  for (auto &&[lhsDim, rhsDim] : llvm::zip_equal(lhsDimsVec, rhsDimsVec)) {
    WaveSymbolAttr lhsExpr = lhs.getShape()[lhsDim];
    WaveSymbolAttr rhsExpr = rhs.getShape()[rhsDim];
    if (lhsExpr == rhsExpr)
      continue;

    return emitError(loc) << "expected " << lhsName << " dimension #" << lhsDim
                          << " (" << lhsExpr << ") to match " << rhsName
                          << " dimension #" << rhsDim << " (" << rhsExpr << ")";
  }
  return success();
}

// Verify that element types of Wave tensors match between LHS and RHS. Emit
// diagnostic errors and return a failure when it is not the case.
static LogicalResult verifyElementTypesMatch(Location loc,
                                             llvm::StringRef lhsName,
                                             WaveTensorType lhs,
                                             llvm::StringRef rhsName,
                                             WaveTensorType rhs) {
  if (lhs.getElementType() == rhs.getElementType())
    return success();

  return emitError(loc) << "expected " << lhsName << " and " << rhsName
                        << " elemental types to match, got "
                        << lhs.getElementType() << ", " << rhs.getElementType();
}

//===----------------------------------------------------------------------===//
// MmaOp
//===----------------------------------------------------------------------===//

LogicalResult MmaOp::verify() {
  WaveTensorType lhsType = getLhs().getType();
  WaveTensorType rhsType = getRhs().getType();
  WaveTensorType accumulatorType = getAccumulator().getType();

  if (failed(
          verifyElementTypesMatch(getLoc(), "LHS", lhsType, "RHS", rhsType)) ||
      failed(verifyElementTypesMatch(getLoc(), "LHS", lhsType, "accumulator",
                                     accumulatorType)))
    return failure();

  return failure(
      verifyTypesMatchingDimensions(getLoc(), "LHS", lhsType, {1}, "RHS",
                                    rhsType, {0})
          .failed() ||
      verifyTypesMatchingDimensions(getLoc(), "LHS", lhsType, {0},
                                    "accumulator", accumulatorType, {0})
          .failed() ||
      verifyTypesMatchingDimensions(getLoc(), "RHS", rhsType, {1},
                                    "accumulator", accumulatorType, {1})
          .failed());
}

//===----------------------------------------------------------------------===//
// RegisterOp
//===----------------------------------------------------------------------===//

LogicalResult RegisterOp::verify() {
  WaveRegisterType resultType = getResult().getType();
  Type elementType = resultType.getElementType();
  Attribute valueAttr = getValue();

  // Verify value attribute compatibility
  auto typedAttr = dyn_cast<TypedAttr>(valueAttr);
  if (!typedAttr) {
    return emitOpError("value attribute (")
           << valueAttr << ") is not compatible with register element type ("
           << elementType << ")";
  }

  Type attrType = typedAttr.getType();
  if (!((attrType == elementType) || (elementType.isIntOrIndexOrFloat() &&
                                      attrType.isIntOrIndexOrFloat()))) {
    return emitOpError("value attribute (")
           << valueAttr << ") is not compatible with register element type ("
           << elementType << ")";
  }
  // TODO: Possibly add tighter restrictions on which types are supported for
  // value

  // Verify index attribute if present
  if (auto indexAttr = getIndexAttr()) {
    // Check that all values in the dictionary are WaveIndexMappingAttr
    for (auto namedAttr : indexAttr.getValue()) {
      if (!isa<WaveIndexMappingAttr>(namedAttr.getValue())) {
        return emitOpError(
                   "index attribute values must be WaveIndexMappingAttr, got ")
               << namedAttr.getValue();
      }

      auto mappingAttr = cast<WaveIndexMappingAttr>(namedAttr.getValue());
      AffineMap map = mappingAttr.getMap();

      // Verify the affine map has no dimensions, only symbols
      if (map.getNumDims() != 0) {
        return emitOpError(
                   "wave indexing affine maps should have no dimensions, only "
                   "symbols, got ")
               << map.getNumDims() << " dimensions for symbol "
               << namedAttr.getName();
      }

      // Check that the symbol name corresponds to a dimension in the register
      // type
      StringRef indexSymbolName = namedAttr.getName().getValue();
      if (!llvm::any_of(resultType.getShape(), [&](auto dimSymbol) {
            return dimSymbol.getName() == indexSymbolName;
          })) {
        return emitOpError("index symbol '")
               << indexSymbolName
               << "' does not correspond to any dimension in register type";
      }
    }
  }

  return success();
}

ParseResult RegisterOp::parse(OpAsmParser &parser, OperationState &result) {
  Attribute valueAttr;
  Type resultType;

  // Parse: '(' value ')'
  if (parser.parseLParen() || parser.parseAttribute(valueAttr) ||
      parser.parseRParen())
    return failure();

  // Parse optional index attribute: 'index' '{' symbol ':' mapping (',' symbol
  // ':' mapping)* '}'
  if (succeeded(parser.parseOptionalKeyword("index"))) {
    SmallVector<NamedAttribute> indexMappings;

    if (failed(parser.parseLBrace()))
      return failure();

    auto parseMapping = [&]() {
      StringRef symbolName;
      if (failed(parser.parseKeyword(&symbolName)) ||
          failed(parser.parseColon()))
        return failure();

      // Try to parse as WaveIndexMappingAttr using custom parsing
      auto indexMapping = WaveIndexMappingAttr::parse(parser, Type{});
      if (!indexMapping) {
        return failure();
      }
      indexMappings.push_back(
          {parser.getBuilder().getStringAttr(symbolName), indexMapping});
      return success();
    };

    if (parser.parseCommaSeparatedList(parseMapping) || parser.parseRBrace())
      return failure();

    result.addAttribute("index",
                        parser.getBuilder().getDictionaryAttr(indexMappings));
  }

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse: ':' result_type
  if (parser.parseColon() || parser.parseType(resultType))
    return failure();

  result.addAttribute("value", valueAttr);
  result.addTypes(resultType);
  return success();
}

void RegisterOp::print(OpAsmPrinter &printer) {
  printer << " (";
  printer.printAttribute(getValue());
  printer << ")";

  // Print index attribute using custom syntax
  if (auto indexAttr = getIndexAttr()) {
    printer << " index {";
    llvm::interleaveComma(
        indexAttr.getValue(), printer, [&](NamedAttribute namedAttr) {
          printer << namedAttr.getName().getValue() << " : ";
          if (auto mappingAttr =
                  dyn_cast<WaveIndexMappingAttr>(namedAttr.getValue())) {
            mappingAttr.print(printer);
          } else {
            printer.printAttribute(namedAttr.getValue());
          }
        });
    printer << "}";
  }

  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"value", "index"});
  printer << " : ";
  printer.printType(getResult().getType());
}
