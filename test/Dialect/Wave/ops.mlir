// RUN: water-opt %s --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @mma
func.func @mma(%lhs: !wave.tensor<[@A, @B] of f32>, %rhs: !wave.tensor<[@B, @C] of f32>, %acc: !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32> {
  // CHECK: wave.mma
  %0 = wave.mma %lhs, %rhs, %acc : (!wave.tensor<[@A, @B] of f32>, !wave.tensor<[@B, @C] of f32>, !wave.tensor<[@A, @C] of f32>) -> !wave.tensor<[@A, @C] of f32>
  return %0 : !wave.tensor<[@A, @C] of f32>
}


// CHECK-LABEL: @register
func.func @register() -> !wave.register<[@M, @N] of f32> {
  // CHECK: wave.register
  %0 = wave.register(0.0) : !wave.register<[@M, @N] of f32>
  return %0 : !wave.register<[@M, @N] of f32>
}

// CHECK-LABEL: @register_i32
func.func @register_i32() -> !wave.register<[@X, @Y] of i32> {
  // CHECK: wave.register
  %0 = wave.register(42) : !wave.register<[@X, @Y] of i32>
  return %0 : !wave.register<[@X, @Y] of i32>
}

// CHECK-LABEL: @register_bf16
func.func @register_bf16() -> !wave.register<[@X, @Y] of bf16> {
  // CHECK: wave.register
  %0 = wave.register(1.5) : !wave.register<[@X, @Y] of bf16>
  return %0 : !wave.register<[@X, @Y] of bf16>
}
