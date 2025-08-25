// RUN: water-opt %s --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @mma
func.func @mma(%lhs: !wave.tensor<[@A, @B] of f16>, %rhs: !wave.tensor<[@B, @C] of f16>, %acc: !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32> {
  // CHECK: wave.mma
  %0 = wave.mma %lhs, %rhs, %acc {kind = #wave.mma_kind<f32_16x16x16_f16>} : (!wave.tensor<[@A, @B] of f16>, !wave.tensor<[@B, @C] of f16>, !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32>
  return %0 : !wave.tensor<[@A, @C] of f32>
}
