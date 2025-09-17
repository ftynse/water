#include "water/Dialect/Wave/IR/WaveUtils.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

std::optional<llvm::SmallVector<int64_t>>
wave::resolveSymbolNames(llvm::ArrayRef<wave::WaveSymbolAttr> names,
                         wave::WaveHyperparameterAttr hyper) {
  if (!hyper)
    return std::nullopt;
  // Collect concrete values for each symbol in stored order.
  llvm::SmallVector<int64_t> symVals;
  symVals.reserve(names.size());
  for (auto symbol : names) {
    auto value = hyper.getSymbolValue(symbol.getName());
    if (!value)
      return std::nullopt;
    symVals.push_back(*value);
  }
  return symVals;
}

std::optional<llvm::SmallVector<int64_t>>
wave::evaluateMapWithSymbols(AffineMap map, llvm::ArrayRef<int64_t> vals) {
  // Build AffineExpr replacements for symbols: s_i → const(symVals[i]).
  MLIRContext *ctx = map.getContext();
  llvm::SmallVector<AffineExpr> symRepls;
  symRepls.reserve(vals.size());
  for (int64_t v : vals)
    symRepls.push_back(getAffineConstantExpr(v, ctx));

  // For each result expr: substitute symbols and fold
  llvm::SmallVector<int64_t> out;
  out.reserve(map.getNumResults());
  for (AffineExpr affine : map.getResults()) {
    AffineExpr sub = affine.replaceSymbols(symRepls);
    sub = simplifyAffineExpr(sub, /*numDims=*/0, /*numSymbols=*/0);
    if (auto c = llvm::dyn_cast<AffineConstantExpr>(sub)) {
      if (c.getValue() < 0)
        return std::nullopt; // optional sanity
      out.push_back(c.getValue());
    } else {
      return std::nullopt;
    }
  }
  return out;
}
