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


// CHECK-LABEL: @register_with_symbols
func.func @register_with_symbols() -> !wave.register<[@M, @N] of f32> {
  // CHECK: wave.register
  %0 = wave.register(0.0)
       index {M : [THREAD_ID, BLOCK_SIZE] -> (THREAD_ID floordiv BLOCK_SIZE),
              N : [THREAD_ID, BLOCK_SIZE] -> (THREAD_ID * BLOCK_SIZE + 42)}
       : !wave.register<[@M, @N] of f32>
  return %0 : !wave.register<[@M, @N] of f32>
}


// CHECK-LABEL: @register_complex_index
func.func @register_complex_index() -> !wave.register<[@B, @N, @M] of f32> {
  // CHECK: wave.register
  %0 = wave.register(0.0)
       index {
         B : [WG2, BLOCK_B] -> (WG2 * BLOCK_B),
         M : [WG0, BLOCK_M, T0] -> (WG0 * BLOCK_M + BLOCK_M * ((T0 floordiv 64) floordiv 2) + T0 mod 32),
         N : [T1, BLOCK_N, WG1, GPR_NUM, T0] -> (T1 * (BLOCK_N floordiv 2) + BLOCK_N * WG1 + GPR_NUM mod 4 + ((GPR_NUM floordiv 4) mod 4) * 8 + ((T0 mod 64) floordiv 32) * 4)
       }
       : !wave.register<[@B, @N, @M] of f32>
  return %0 : !wave.register<[@B, @N, @M] of f32>
}
