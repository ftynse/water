// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::water {
#define GEN_PASS_DEF_WATERASSERTINBOUNDSPASS
#include "water/Transforms/Passes.h.inc"
} // namespace mlir::water

using namespace mlir;

/// Kind of checks to insert.
enum class CheckKind { PerDimension, Combined };

// These should become an interface eventually, but currently not worth the
// complexity of adding one.

/// Whether the operation is supposed by the assertion inserter.
static bool isSupported(memref::LoadOp) { return true; }
static bool isSupported(memref::StoreOp) { return true; }
static bool isSupported(vector::LoadOp loadOp) {
  return !loadOp.getVectorType().isScalable();
}
static bool isSupported(vector::StoreOp storeOp) {
  return !storeOp.getVectorType().isScalable();
}

/// Returns the number of elements accessed along the given dimension by a
/// vector load/store operation.
template <typename OpTy>
static int64_t getAccessExtentAlongDimVectorOp(OpTy op, unsigned dim) {
  static_assert(llvm::is_one_of<OpTy, vector::LoadOp, vector::StoreOp>::value,
                "expected vector load or store");
  if (isa<VectorType>(op.getMemRefType().getElementType()))
    return 1;
  unsigned leadingOneDims =
      op.getMemRefType().getRank() - op.getVectorType().getRank();
  return dim < leadingOneDims
             ? 1
             : op.getVectorType().getDimSize(dim - leadingOneDims);
}

/// Returns the number of elements accessed along the given dimension by the
/// operation.
static int64_t getAccessExtentAlongDim(memref::LoadOp, unsigned) { return 1; }
static int64_t getAccessExtentAlongDim(memref::StoreOp, unsigned) { return 1; }
static int64_t getAccessExtentAlongDim(vector::LoadOp loadOp, unsigned dim) {
  return getAccessExtentAlongDimVectorOp(loadOp, dim);
}
static int64_t getAccessExtentAlongDim(vector::StoreOp storeOp, unsigned dim) {
  return getAccessExtentAlongDimVectorOp(storeOp, dim);
}

/// Returns the base memref that is being indexed into by the accessing
/// operation.
static Value getAccessBase(memref::LoadOp loadOp) { return loadOp.getMemRef(); }
static Value getAccessBase(memref::StoreOp storeOp) {
  return storeOp.getMemRef();
}
static Value getAccessBase(vector::LoadOp loadOp) { return loadOp.getBase(); }
static Value getAccessBase(vector::StoreOp storeOp) {
  return storeOp.getBase();
}

// End pseudo-interface.

/// Inserts `cf.assert` checking whether the subscripts of the given
/// memory-accessing operation are in bounds.
template <typename OpTy>
static LogicalResult insertInBoundsAssertions(OpBuilder &builder, OpTy op,
                                              CheckKind checkKind) {
  if (!isSupported(op))
    return op.emitError() << "unsupported variation of the op";

  ImplicitLocOpBuilder b(op->getLoc(), builder);
  Value zero = b.createOrFold<arith::ConstantIndexOp>(0);
  Value totalCheck =
      checkKind == CheckKind::Combined
          ? b.createOrFold<arith::ConstantIntOp>(1, b.getI1Type())
          : nullptr;
  for (unsigned i = 0, e = op.getMemRefType().getRank(); i < e; ++i) {
    Value index = b.createOrFold<arith::ConstantIndexOp>(i);
    Value dim = b.createOrFold<memref::DimOp>(getAccessBase(op), index);
    Value subscript = op.getIndices()[i];
    Value lowerBoundCheck = b.createOrFold<arith::CmpIOp>(
        arith::CmpIPredicate::sge, subscript, zero);

    int64_t accessExtent = getAccessExtentAlongDim(op, i);
    assert(accessExtent >= 1 && "expected positive access extent");
    Value lastAccessedIndex =
        accessExtent == 1
            ? subscript
            : b.createOrFold<arith::AddIOp>(
                  subscript,
                  b.createOrFold<arith::ConstantIndexOp>(accessExtent - 1));

    Value upperBoundCheck = b.createOrFold<arith::CmpIOp>(
        arith::CmpIPredicate::slt, lastAccessedIndex, dim);
    Value boundCheck =
        b.createOrFold<arith::AndIOp>(lowerBoundCheck, upperBoundCheck);
    if (checkKind == CheckKind::PerDimension) {
      b.createOrFold<cf::AssertOp>(
          boundCheck,
          "memref access out of bounds along dimension " + std::to_string(i));
    } else {
      assert(checkKind == CheckKind::Combined && "unsupported check kind");
      totalCheck = b.createOrFold<arith::AndIOp>(totalCheck, boundCheck);
    }
  }
  if (checkKind == CheckKind::Combined) {
    b.createOrFold<cf::AssertOp>(totalCheck, "memref access out of bounds");
  }
  return success();
}
namespace {
class AssertInBoundsPass
    : public water::impl::WaterAssertInBoundsPassBase<AssertInBoundsPass> {
public:
  using WaterAssertInBoundsPassBase::WaterAssertInBoundsPassBase;

  void runOnOperation() override;
};

void AssertInBoundsPass::runOnOperation() {
  CheckKind checkKind =
      checkEachDim ? CheckKind::PerDimension : CheckKind::Combined;

  OpBuilder builder(&getContext());
  WalkResult walkResult = getOperation()->walk([&](Operation *op) {
    OpBuilder::InsertionGuard raii(builder);
    builder.setInsertionPoint(op);
    LogicalResult result =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case<memref::LoadOp, memref::StoreOp>([&](auto casted) {
              return insertInBoundsAssertions(builder, casted, checkKind);
            })
            .Case<vector::LoadOp, vector::StoreOp>([&](auto casted) {
              // Vector load/store specifically allow for lowering-defined
              // out-of-bounds access when using scalar-typed memory. Ignore
              // those unless explicitly requested by the caller.
              if (!includeVectorLoadStore &&
                  !isa<VectorType>(casted.getMemRefType().getElementType()))
                return success();

              return insertInBoundsAssertions(builder, casted, checkKind);
            })
            .Default([&](Operation *uncasted) {
              if (!warnOnUnknown)
                return success();

              auto effecting = dyn_cast<MemoryEffectOpInterface>(uncasted);
              if (!effecting)
                return success();

              SmallVector<MemoryEffects::EffectInstance> effects;
              effecting.getEffects(effects);
              if (llvm::none_of(
                      effects, [](MemoryEffects::EffectInstance &instance) {
                        bool effectMayBeOnMemRef =
                            !instance.getValue() ||
                            isa<MemRefType>(instance.getValue().getType());
                        return effectMayBeOnMemRef &&
                               isa<MemoryEffects::Read, MemoryEffects::Write>(
                                   instance.getEffect());
                      }))
                return success();

              uncasted->emitWarning()
                  << "operation with memory effects was not processed";
              return success();
            });
    if (failed(result))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    signalPassFailure();
}
} // namespace
