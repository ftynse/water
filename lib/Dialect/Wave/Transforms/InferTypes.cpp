// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "wave-infer-types"

namespace wave {
#define GEN_PASS_DEF_WATERWAVEINFERTYPESPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

namespace {

class InferTypeLatticeStorage {
public:
  InferTypeLatticeStorage() : value(nullptr, kUndecidableState) {}
  InferTypeLatticeStorage(const InferTypeLatticeStorage &value) = default;
  InferTypeLatticeStorage(wave::WaveTensorType concreteValue)
      : value(concreteValue, kSpecificTypeState) {}

  InferTypeLatticeStorage &
  operator=(const InferTypeLatticeStorage &other) = default;

  bool operator==(const InferTypeLatticeStorage &other) const {
    return value == other.value;
  }

  bool isBottom() const { return value.getInt() == kUninitializedState; }

  bool isTop() const { return value.getInt() == kUndecidableState; }

  wave::WaveTensorType getConcreteValue() const {
    if (value.getInt() != kSpecificTypeState)
      return nullptr;
    return llvm::cast<wave::WaveTensorType>(value.getPointer());
  }

  static InferTypeLatticeStorage top() {
    InferTypeLatticeStorage result;
    result.value.setPointer(nullptr);
    result.value.setInt(kUninitializedState);
    return result;
  }

  static InferTypeLatticeStorage join(const InferTypeLatticeStorage &lhs,
                                      const InferTypeLatticeStorage &rhs) {
    if (lhs.value == rhs.value)
      return lhs;

    if (lhs.isBottom())
      return rhs;

    if (rhs.isBottom())
      return lhs;

    return top();
  }

  void unsafeSet(const InferTypeLatticeStorage &value) {
    this->value = value.value;
  }

  void print(llvm::raw_ostream &os) const {
    if (isBottom())
      os << "<bottom>";
    else if (isTop())
      os << "<top>";
    else
      os << getConcreteValue();
  }

private:
  llvm::PointerIntPair<mlir::Type, 2> value;

  const static unsigned kUninitializedState = 0;
  const static unsigned kSpecificTypeState = 1;
  const static unsigned kUndecidableState = 2;
};

class InferTypeLattice
    : public mlir::dataflow::Lattice<InferTypeLatticeStorage> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferTypeLattice);
  using Lattice::Lattice;
};

class InferTypeForwardAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<InferTypeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  mlir::LogicalResult initialize(mlir::Operation *top) override {
    top->walk([&](mlir::Operation *op) {
      if (auto iface = llvm::dyn_cast<wave::WaveInferTypeOpInterface>(op)) {
        for (mlir::Value result : iface->getResults()) {
          InferTypeLattice *lattice = initForValue(result);
          if (!lattice)
            continue;
          addDependency(lattice, getProgramPointBefore(iface));
        }
        return mlir::WalkResult::advance();
      } else if (auto iface = llvm::dyn_cast<mlir::FunctionOpInterface>(op)) {
        if (iface.isDeclaration())
          return mlir::WalkResult::advance();

        for (mlir::Value arg : iface.getFunctionBody().front().getArguments()) {
          initForValue(arg);
        }
      }
      return mlir::WalkResult::advance();
    });
    return mlir::success();
  }

  void setToEntryState(InferTypeLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(InferTypeLatticeStorage::top()));
  }

  mlir::LogicalResult
  visitOperation(mlir::Operation *op,
                 llvm::ArrayRef<const InferTypeLattice *> operands,
                 llvm::ArrayRef<InferTypeLattice *> results) override {

    llvm::errs() << "visiting " << *op << "\n";

    auto iface = llvm::dyn_cast<wave::WaveInferTypeOpInterface>(op);
    if (!iface) {
      return op->emitError()
             << "cannot propagate types across an operation not implementing "
                "the wave infer type interface";
    }

    auto extractType = [](const InferTypeLattice *lattice) {
      return lattice->getValue().getConcreteValue();
    };
    llvm::SmallVector<wave::WaveTensorType> operandTypes =
        llvm::map_to_vector(operands, extractType);
    llvm::SmallVector<wave::WaveTensorType> resultTypes =
        llvm::map_to_vector(results, extractType);

    std::string errorMessage;
    llvm::raw_string_ostream errs(errorMessage);
    llvm::FailureOr<mlir::ChangeResult> result =
        iface.propagateForward(operandTypes, resultTypes, errs);
    if (mlir::failed(result)) {
      return op->emitError()
             << "failed to propagate type information forward: " << errs.str();
    }
    if (*result == mlir::ChangeResult::NoChange)
      return mlir::success();

    for (auto &&[result, lattice] : llvm::zip_equal(resultTypes, results)) {
      propagateIfChanged(lattice,
                         lattice->join(InferTypeLatticeStorage(result)));
    }
    return mlir::success();
  }

private:
  InferTypeLattice *initForValue(mlir::Value value) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(value.getType());
    if (!tensorType)
      return nullptr;
    InferTypeLattice *lattice = getLatticeElement(value);
    lattice->getValue().unsafeSet(InferTypeLatticeStorage(tensorType));
    propagateIfChanged(lattice, mlir::ChangeResult::Change);
    return lattice;
  }
};

class InferTypes : public wave::impl::WaterWaveInferTypesPassBase<InferTypes> {
public:
  void runOnOperation() override {
    mlir::DataFlowConfig dataFlowConfig;
    dataFlowConfig.setInterprocedural(false);
    mlir::DataFlowSolver solver(dataFlowConfig);
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<InferTypeForwardAnalysis>();
    mlir::Operation *root = getOperation();
    if (mlir::failed(solver.initializeAndRun(root))) {
      getOperation()->emitError() << "dataflow analysis failed";
      return signalPassFailure();
    }

    auto updateType = [&](mlir::Value value, llvm::StringRef description) {
      if (!llvm::isa<wave::WaveTensorType>(value.getType()))
        return mlir::success();

      auto *lattice = solver.lookupState<InferTypeLattice>(value);
      if (!lattice || lattice->getValue().isBottom()) {
        emitError(value.getLoc()) << "couldn't infer type for " << description;
        return mlir::failure();
      }
      if (lattice->getValue().isTop()) {
        emitError(value.getLoc())
            << "type conflict was detected for " << description;
        return mlir::failure();
      }

      value.setType(lattice->getValue().getConcreteValue());
      return mlir::success();
    };

    mlir::WalkResult walkResult =
        getOperation()->walk([&](mlir::Operation *op) {
          for (mlir::OpResult res : op->getResults()) {
            if (mlir::failed(updateType(
                    res, "result #" + std::to_string(res.getResultNumber()))))
              return mlir::WalkResult::interrupt();
          }

          for (mlir::Region &region : op->getRegions()) {
            for (auto &&[blockNumber, block] : llvm::enumerate(region)) {
              for (mlir::BlockArgument arg : block.getArguments()) {
                auto fmt = llvm::formatv(
                    "argument #{0} of block #{1} in region #{2}",
                    region.getRegionNumber(), blockNumber, arg.getArgNumber());
                if (mlir::failed(updateType(arg, fmt.str())))
                  return mlir::WalkResult::interrupt();
              }
            }
          }

          return mlir::WalkResult::advance();
        });

    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace
