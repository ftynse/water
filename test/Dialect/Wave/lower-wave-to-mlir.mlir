// RUN: water-opt %s -allow-unregistered-dialect -lower-wave-to-mlir -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @lower_register
func.func @lower_register() attributes {wave.hyperparameters = #wave.hyperparameters<{X = 100, Y = 10, Z = 1}>} {
  // CHECK-NOT: wave.register
  // CHECK:     arith.constant dense<0.000000e+00> : vector<10x1xf32>
  %cst = arith.constant 0.0 : f32
  wave.register %cst : !wave.tensor<[@Y, @Z] of f32, <register>>
  %cst1 = arith.constant 1.0 : f32
  // CHECK:     arith.constant dense<1.000000e+00> : vector<10x1x100xf32>
  wave.register %cst1 : !wave.tensor<[@Y, @Z, @X] of f32, <register>>
  %cst2 = arith.constant 2 : i32
  // CHECK:     arith.constant dense<2> : vector<10x1x100xi32>
  wave.register %cst2 : !wave.tensor<[@Y, @Z, @X] of i32, <register>>
  return
}

// -----

// CHECK-LABEL: func.func @lower_register_with_unrealized_cast
func.func @lower_register_with_unrealized_cast() attributes {wave.hyperparameters = #wave.hyperparameters<{X = 100, Y = 10, Z = 1}>} {
  // CHECK-NOT: wave.register
  // CHECK:     %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<10x1xf32>
  // CHECK:     %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[CST]] : vector<10x1xf32> to !wave.tensor<[@Y, @Z] of f32, <register>>
  // CHECK:     "test.foo"(%[[CAST]])
  %cst = arith.constant 0.0 : f32
  %0 = wave.register %cst : !wave.tensor<[@Y, @Z] of f32, <register>>
  "test.foo"(%0) : (!wave.tensor<[@Y, @Z] of f32, <register>>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @lower_nested_register
func.func @lower_nested_register(%cond: i1) attributes {wave.hyperparameters = #wave.hyperparameters<{X = 100, Y = 10, Z = 1}>} {
  // CHECK-NOT: wave.register
  scf.if %cond {
    %cst = arith.constant 0.0 : f32
    // CHECK:     arith.constant dense<0.000000e+00> : vector<10x1xf32>
    wave.register %cst : !wave.tensor<[@Y, @Z] of f32, <register>>
    scf.yield
  } else {
    %cst1 = arith.constant 1.0 : f32
    // CHECK:     arith.constant dense<1.000000e+00> : vector<10x1x100xf32>
    wave.register %cst1 : !wave.tensor<[@Y, @Z, @X] of f32, <register>>
    scf.yield
  }
  return
}
