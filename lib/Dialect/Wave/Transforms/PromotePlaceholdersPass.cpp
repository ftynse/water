// Copyright 2025 The Water Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://www.llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "water/Dialect/Wave/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/IR/WaveTypes.h"

using namespace mlir;

namespace {

struct PromotePlaceholdersPass
    : public wave::impl::PromotePlaceholdersPassBase<PromotePlaceholdersPass> {
  using PromotePlaceholdersPassBase::PromotePlaceholdersPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();
    IRRewriter rewriter(ctx);

    DenseMap<Value, Value> promoted;

    for (Block &block : func) {
      for (Operation &op : llvm::make_early_inc_range(block)) {

        if (auto read = dyn_cast<wave::ReadOp>(&op)) {
          Value srcMem = read.getMemory();
          auto srcTy = llvm::dyn_cast<wave::WaveTensorType>(srcMem.getType());
          if (!srcTy)
            continue;

          if (srcTy.getAddressSpaceValue() != wave::WaveAddressSpace::Shared)
            continue;

          // Reuse or create a shared alloc for this source mem
          Value sharedBuf = promoted.lookup(srcMem);
          if (!sharedBuf) {
            SmallVector<Attribute> distElems;
            distElems.reserve(srcTy.getShape().size());
            for (wave::WaveSymbolAttr s : srcTy.getShape())
              distElems.push_back(s);
            ArrayAttr dist = ArrayAttr::get(ctx, distElems);

            {
              IRRewriter::InsertionGuard g(rewriter);
              rewriter.setInsertionPoint(read);

              auto sharedAS = wave::WaveAddressSpaceAttr::get(
                  ctx, wave::WaveAddressSpace::Shared);
              auto sharedTy = wave::WaveTensorType::get(
                  ctx, srcTy.getShape(), srcTy.getFullySpecified(),
                  srcTy.getElementType(), sharedAS);

              auto alloc = rewriter.create<wave::AllocateOp>(read.getLoc(),
                                                             sharedTy, dist);
              sharedBuf = alloc.getResult();
              promoted[srcMem] = sharedBuf;

              // seed contents: cast the handle to global, read(global) →
              // write(shared).
              auto globalAS = wave::WaveAddressSpaceAttr::get(
                  ctx, wave::WaveAddressSpace::Global);
              auto globalTy = wave::WaveTensorType::get(
                  ctx, srcTy.getShape(), srcTy.getFullySpecified(),
                  srcTy.getElementType(), globalAS);

              auto castOp = rewriter.create<wave::AddrSpaceCastOp>(
                  read.getLoc(), globalTy, srcMem, globalAS);
              Value globalMem = castOp.getResult();

              auto regVal =
                  rewriter.create<wave::ReadOp>(read.getLoc(), srcTy, globalMem)
                      .getResult();
              (void)rewriter.create<wave::WriteOp>(read.getLoc(), regVal,
                                                   sharedBuf);
            }
          }

          // Redirect the current read to come from shared alloc
          {
            IRRewriter::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(read);
            auto newRead = rewriter.create<wave::ReadOp>(
                read.getLoc(), read.getResult().getType(), sharedBuf);
            rewriter.replaceOp(read, newRead.getResult());
          }
          continue;
        }

        if (auto write = dyn_cast<wave::WriteOp>(&op)) {
          Value dstMem = write.getMemory();
          auto dstTy = llvm::dyn_cast<wave::WaveTensorType>(dstMem.getType());
          if (!dstTy)
            continue;

          if (dstTy.getAddressSpaceValue() != wave::WaveAddressSpace::Shared)
            continue;

          Value sharedBuf = promoted.lookup(dstMem);
          if (!sharedBuf) {
            SmallVector<Attribute> distElems;
            distElems.reserve(dstTy.getShape().size());
            for (wave::WaveSymbolAttr s : dstTy.getShape())
              distElems.push_back(s);
            ArrayAttr dist = ArrayAttr::get(ctx, distElems);

            IRRewriter::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(write);

            auto sharedAS = wave::WaveAddressSpaceAttr::get(
                ctx, wave::WaveAddressSpace::Shared);
            auto sharedTy = wave::WaveTensorType::get(
                ctx, dstTy.getShape(), dstTy.getFullySpecified(),
                dstTy.getElementType(), sharedAS);

            auto alloc = rewriter.create<wave::AllocateOp>(write.getLoc(),
                                                           sharedTy, dist);
            sharedBuf = alloc.getResult();
            promoted[dstMem] = sharedBuf;
          }

          {
            IRRewriter::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(write);
            auto newWrite = rewriter.create<wave::WriteOp>(
                write.getLoc(), write.getValue(), sharedBuf);
            rewriter.replaceOp(write, newWrite);
          }
          continue;
        }
      }
    }
  }
};

} // namespace

#define GEN_PASS_DEF_PROMOTEPLACEHOLDERS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> wave::createPromotePlaceholdersPass() {
  return std::make_unique<PromotePlaceholdersPass>();
}
