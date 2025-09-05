// RUN: water-opt %s -allow-unregistered-dialect -lower-wave-to-mlir -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @lower_register
func.func @lower_register() {
  // CHECK-NOT: wave.register
  // CHECK:     arith.constant dense<0.000000e+00> : vector<9x9xf32>
  %cst = arith.constant 0.0 : f32
  wave.register %cst : !wave.tensor<[@Y, @Z] of f32, <register>>
  %cst1 = arith.constant 1.0 : f32
  // CHECK:     arith.constant dense<1.000000e+00> : vector<9x9x9xf32>
  wave.register %cst1 : !wave.tensor<[@Y, @Z, @X] of f32, <register>>
  return
}

// -----

// CHECK-LABEL: func.func @lower_register_with_unrealized_cast
func.func @lower_register_with_unrealized_cast() {
  // CHECK-NOT: wave.register
  // CHECK:     %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<9x9xf32>
  // CHECK:     %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[CST]] : vector<9x9xf32> to !wave.tensor<[@Y, @Z] of f32, <register>>
  // CHECK:     "test.foo"(%[[CAST]])
  %cst = arith.constant 0.0 : f32
  %0 = wave.register %cst : !wave.tensor<[@Y, @Z] of f32, <register>>
  "test.foo"(%0) : (!wave.tensor<[@Y, @Z] of f32, <register>>) -> ()
  return
}
