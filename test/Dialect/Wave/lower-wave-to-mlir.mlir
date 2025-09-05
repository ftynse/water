// RUN: water-opt %s -lower-wave-to-mlir -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @lower_register
func.func @lower_register() {
  // CHECK-NOT: wave.register
  // CHECK:     arith.constant dense<0.000000e+00> : vector<9x9xf32>
  %cst = arith.constant 0.0 : f32
  wave.register %cst : !wave.tensor<[@Y, @Z] of f32>
  %cst1 = arith.constant 1.0 : f32
  // CHECK:     arith.constant dense<1.000000e+00> : vector<9x9x9xf32>
  wave.register %cst1 : !wave.tensor<[@Y, @Z, @X] of f32>
  return
}
