// RUN: water-opt %s --water-assert-in-bounds --split-input-file | FileCheck %s --check-prefixes=CHECK,PERDIM
// RUN: water-opt %s --water-assert-in-bounds='check-each-dim=false' --split-input-file | FileCheck %s --check-prefixes=CHECK,COMPOUND
// RUN: water-opt %s --water-assert-in-bounds='include-vector-load-store=true' --split-input-file | FileCheck %s --check-prefixes=CHECK,VECTOR

// CHECK-LABEL: @all_static
func.func @all_static(%memref: memref<2x2xf32>) -> f32 {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  // PERDIM:         %[[TRUE:.+]] = arith.constant true
  // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
  // PERDIM:         %[[TRUE1:.+]] = arith.constant true
  // PERDIM:         cf.assert %[[TRUE1]], "memref access out of bounds along dimension 1"
  //
  // COMPOUND:       %[[TRUE:.+]] = arith.constant true
  // COMPOUND:       cf.assert %[[TRUE]], "memref access out of bounds"
  // CHECK:          memref.load
  %2 = memref.load %memref[%0, %1] : memref<2x2xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: @shape_static_one_index_dynamic
// CHECK-SAME: (%[[ARG0:.+]]: memref<5x2xf32>, %[[ARG1:.+]]: index)
func.func @shape_static_one_index_dynamic(%memref: memref<5x2xf32>, %i: index) -> f32 {
  // CHECK:          arith.constant 0
  %0 = arith.constant 0 : index
  // CHECK:          %[[ZERO:.+]] = arith.constant 0 : index
  // CHECK:          %[[TRUE:.+]] = arith.constant true
  // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
  //
  // CHECK:          %[[TWO:.+]] = arith.constant 2 : index
  // CHECK:          %[[LB:.+]] = arith.cmpi sge, %[[ARG1]], %[[ZERO]]
  // CHECK:          %[[UB:.+]] = arith.cmpi slt, %[[ARG1]], %[[TWO]]
  // CHECK:          %[[BOUND:.+]] = arith.andi %[[LB]], %[[UB]]
  // PERDIM:         cf.assert %[[BOUND]], "memref access out of bounds along dimension 1"
  //
  // COMPOUND:       %[[COMPOUND:.+]] = arith.andi %[[BOUND]], %[[TRUE]]
  // COMPOUND:       cf.assert %[[COMPOUND]], "memref access out of bounds"
  // CHECK:          memref.load
  %1 = memref.load %memref[%0, %i] : memref<5x2xf32>
  return %1 : f32
}

// -----

// CHECK-LABEL: @shape_dynamic
func.func @shape_dynamic(%memref: memref<?x?xf32>) -> f32 {
  // CHECK:          %[[INDEX0:.+]] = arith.constant 0
  // CHECK:          %[[INDEX1:.+]] = arith.constant 1
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  // CHECK:          %[[ZERO1:.+]] = arith.constant 0 : index
  // CHECK:          %[[ZERO2:.+]] = arith.constant 0 : index
  // CHECK:          %[[DIM0:.+]] = memref.dim %{{.*}}, %[[ZERO2]]
  // Note that folding changed index0 < dim0 into dim0 > index0.
  // CHECK:          %[[UB0:.+]] = arith.cmpi sgt, %[[DIM0]], %[[INDEX0]]
  // PERDIM:         %[[BOUND0:.+]] = arith.andi %[[UB0]]
  // PERDIM:         cf.assert %[[BOUND0]], "memref access out of bounds along dimension 0"
  // COMPOUND:       %[[PREBOUND0:.+]] = arith.andi %[[UB0]]
  // COMPOUND:       %[[BOUND0:.+]] = arith.andi %[[PREBOUND0]]
  //
  // CHECK:          %[[ONE1:.+]] = arith.constant 1 : index
  // CHECK:          %[[DIM1:.+]] = memref.dim %{{.*}}, %[[ONE1]]
  // CHECK:          %[[UB1:.+]] = arith.cmpi sgt, %[[DIM1]], %[[INDEX1]]
  // CHECK:          %[[BOUND1:.+]] = arith.andi %[[UB1]]
  // PERDIM:         cf.assert %[[BOUND1]], "memref access out of bounds along dimension 1"
  //
  // COMPOUND:       %[[COMPOUND:.+]] = arith.andi %[[BOUND0]], %[[BOUND1]]
  // COMPOUND:       cf.assert %[[COMPOUND]], "memref access out of bounds"
  //
  // CHECK:          memref.load
  %2 = memref.load %memref[%0, %1] : memref<?x?xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: @out_of_bounds
func.func @out_of_bounds(%memref: memref<2x2xf32>) -> f32 {
  %0 = arith.constant 0 : index
  %1 = arith.constant 2 : index
  // Note that even though the access still hits allocated memory, it wraps
  // around the innermost dimension and is therefore considered out of bounds.
  //
  // PERDIM:         %[[TRUE:.+]] = arith.constant true
  // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
  // CHECK:          %[[FALSE:.+]] = arith.constant false
  // PERDIM:         cf.assert %[[FALSE]], "memref access out of bounds along dimension 1"
  // COMPOUND:       cf.assert %[[FALSE]], "memref access out of bounds"
  // CHECK:          memref.load
  %2 = memref.load %memref[%0, %1] : memref<2x2xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: @zero_d
func.func @zero_d(%memref: memref<f32>) -> f32 {
  // CHECK: memref.load
  %0 = memref.load %memref[] : memref<f32>
  return %0 : f32
}

// -----

// CHECK-LABEL: @vector_load_vector_static
func.func @vector_load_vector_static(%memref: memref<2x2xvector<2xf32>>) -> vector<2xf32> {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  // PERDIM:         %[[TRUE:.+]] = arith.constant true
  // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
  // PERDIM:         %[[TRUE1:.+]] = arith.constant true
  // PERDIM:         cf.assert %[[TRUE1]], "memref access out of bounds along dimension 1"
  //
  // COMPOUND:       %[[TRUE:.+]] = arith.constant true
  // COMPOUND:       cf.assert %[[TRUE]], "memref access out of bounds"
  // CHECK:          vector.load
  %2 = vector.load %memref[%0, %1] : memref<2x2xvector<2xf32>>, vector<2xf32>
  return %2 : vector<2xf32>
}

// -----

// CHECK-LABEL: @vector_load_scalar_static_1d_in_bounds
func.func @vector_load_scalar_static_1d_in_bounds(%memref: memref<5x9xf32>) -> vector<2xf32> {
  %0 = arith.constant 0 : index
  // PERDIM-NOT:     cf.assert
  // COMPOUND-NOT:   cf.assert
  // VECTOR:         %[[TRUE0:.+]] = arith.constant true
  // VECTOR:         cf.assert %[[TRUE0]], "memref access out of bounds along dimension 0"
  // VECTOR:         %[[TRUE1:.+]] = arith.constant true
  // VECTOR:         cf.assert %[[TRUE1]], "memref access out of bounds along dimension 1"
  // CHECK:          vector.load
  %2 = vector.load %memref[%0, %0] : memref<5x9xf32>, vector<2xf32>
  return %2 : vector<2xf32>
}


// -----

// CHECK-LABEL: @vector_load_scalar_static_1d_out_of_bounds
func.func @vector_load_scalar_static_1d_out_of_bounds(%memref: memref<5x9xf32>) -> vector<2xf32> {
  %0 = arith.constant 0 : index
  %1 = arith.constant 8 : index
  // PERDIM-NOT:     cf.assert
  // COMPOUND-NOT:   cf.assert
  // VECTOR:         %[[TRUE:.+]] = arith.constant true
  // VECTOR:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
  // VECTOR:         %[[FALSE:.+]] = arith.constant false
  // VECTOR:         cf.assert %[[FALSE]], "memref access out of bounds along dimension 1"
  // CHECK:          vector.load
  %2 = vector.load %memref[%0, %1] : memref<5x9xf32>, vector<2xf32>
  return %2 : vector<2xf32>
}

// -----

// From this point, we only check the PERDIM scenario as checking dimensions
// separately or together has no incidence on how the control flow is handled.

// CHECK-LABEL: @loop
func.func @loop(%memref: memref<2xf32>, %lb: index, %ub: index, %step: index) -> f32 {
  %0 = arith.constant 0.0 : f32
  // PERDIM:         scf.for %[[I:.+]] = %{{.*}} to
  %1 = scf.for %i = %lb to %ub step %step iter_args(%acc = %0) -> (f32) {
    // PERDIM:         %[[ZERO:.+]] = arith.constant 0 : index
    // PERDIM:         %[[TWO:.+]] = arith.constant 2 : index
    // PERDIM:         %[[LB:.+]] = arith.cmpi sge, %[[I]], %[[ZERO]] : index
    // PERDIM:         %[[UB:.+]] = arith.cmpi slt, %[[I]], %[[TWO]] : index
    // PERDIM:         %[[BOUND:.+]] = arith.andi %[[LB]], %[[UB]]
    // PERDIM:         cf.assert %[[BOUND]], "memref access out of bounds along dimension 0"
    %2 = memref.load %memref[%i] : memref<2xf32>
    %3 = arith.addf %acc, %2 : f32
    scf.yield %3 : f32
  }
  return %1 : f32
}

// -----

// CHECK-LABEL: @loop
func.func @loop_store(%memref: memref<2xf32>, %lb: index, %ub: index, %step: index) {
  // PERDIM:         scf.for %[[I:.+]] = %{{.*}} to
  scf.for %i = %lb to %ub step %step {
    // PERDIM:         %[[ZERO:.+]] = arith.constant 0 : index
    // PERDIM:         %[[TWO:.+]] = arith.constant 2 : index
    // PERDIM:         %[[LB:.+]] = arith.cmpi sge, %[[I]], %[[ZERO]] : index
    // PERDIM:         %[[UB:.+]] = arith.cmpi slt, %[[I]], %[[TWO]] : index
    // PERDIM:         %[[BOUND:.+]] = arith.andi %[[LB]], %[[UB]]
    // PERDIM:         cf.assert %[[BOUND]], "memref access out of bounds along dimension 0"
    %2 = arith.index_cast %i : index to i64
    %3 = arith.sitofp %2 : i64 to f32
    memref.store %3, %memref[%i] : memref<2xf32>
    scf.yield
  }
  return
}

// -----

// CHECK-LABEL: @if
func.func @if(%memref: memref<2xf32>, %val: f32, %cond: i1){
  // PERDIM:         scf.if
  scf.if %cond {
    // PERDIM:         %[[TRUE:.+]] = arith.constant true
    // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
    %1 = arith.constant 0 : index
    memref.store %val, %memref[%1] : memref<2xf32>
  }
  return
}

// -----

// CHECK-LABEL: @if_else
func.func @if_else(%memref: memref<2xf32>, %cond: i1) -> f32 {
  // PERDIM:         scf.if
  %0 = scf.if %cond -> f32 {
    %1 = arith.constant 0 : index
    // PERDIM:         %[[TRUE:.+]] = arith.constant true
    // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
    %2 = memref.load %memref[%1] : memref<2xf32>
    scf.yield %2 : f32
  // PERDIM:         else
  } else {
    %3 = arith.constant 0 : index
    // PERDIM:         %[[TRUE:.+]] = arith.constant true
    // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
    %4 = memref.load %memref[%3] : memref<2xf32>
    scf.yield %4 : f32
  }
  return %0 : f32
}

// -----

// CHECK-LABEL: @if_else_store
func.func @if_else_store(%memref: memref<2xf32>, %cond: i1, %value: f32) {
  // PERDIM:         scf.if
  scf.if %cond {
    %0 = arith.constant 0 : index
    // PERDIM:         %[[TRUE:.+]] = arith.constant true
    // PERDIM:         cf.assert %[[TRUE]], "memref access out of bounds along dimension 0"
    memref.store %value, %memref[%0] : memref<2xf32>
  } else {
    %1 = arith.constant 2 : index
    // PERDIM:         %[[FALSE:.+]] = arith.constant false
    // PERDIM:         cf.assert %[[FALSE]], "memref access out of bounds along dimension 0"
    memref.store %value, %memref[%1] : memref<2xf32>
  }
  return
}

// -----

// CHECK-LABEL: @nested_for_indep
func.func @nested_for_indep(%memref: memref<?xf32>, %lb: index, %ub: index, %step: index) {
  // PERDIM:         scf.for %[[I:.+]] = %{{.*}} to
  scf.for %i = %lb to %ub step %step {
    %c1 = arith.constant 1 : index
    %lb1 = arith.addi %i, %c1 : index
    // PERDIM:         scf.for %[[J:.+]] = %{{.*}} to
    scf.for %j = %lb1 to %ub step %step {
      %sum = arith.addi %i, %j : index
      %1 = arith.index_cast %sum : index to i64
      %2 = arith.uitofp %1 : i64 to f32
      // PERDIM:         %[[ZERO0:.+]] = arith.constant 0
      // PERDIM:         %[[ZERO1:.+]] = arith.constant 0
      // PERDIM:         %[[DIM0:.+]] = memref.dim %{{.*}}, %[[ZERO1]]
      // PERDIM:         %[[LB:.+]] = arith.cmpi sge, %[[J]], %[[ZERO0]]
      // PERDIM:         %[[UB:.+]] = arith.cmpi slt, %[[J]], %[[DIM0]]
      // PERDIM:         %[[BOUND:.+]] = arith.andi %[[LB]], %[[UB]]
      // PERDIM:         cf.assert %[[BOUND]], "memref access out of bounds along dimension 0"
      // CHECK:          memref.store
      memref.store %2, %memref[%j] : memref<?xf32>
    }
  }
  return
}

// -----

// CHECK-LABEL: @nested_for
func.func @nested_for(%memref: memref<?x?xf32>, %lb: index, %ub: index, %step: index) {
  // PERDIM:         scf.for %[[I:.+]] = %{{.*}} to
  scf.for %i = %lb to %ub step %step {
    %c1 = arith.constant 1 : index
    %lb1 = arith.addi %i, %c1 : index
    // PERDIM:         scf.for %[[J:.+]] = %{{.*}} to
    scf.for %j = %lb1 to %ub step %step iter_args(%secondary = %i) -> index {
      %sum = arith.addi %secondary, %c1 : index
      %1 = arith.index_cast %sum : index to i64
      %2 = arith.uitofp %1 : i64 to f32
      // PERDIM:         %[[ZERO0:.+]] = arith.constant 0
      // PERDIM:         %[[ZERO1:.+]] = arith.constant 0
      // PERDIM:         %[[DIM0:.+]] = memref.dim %{{.*}}, %[[ZERO1]]
      // PERDIM:         %[[LB0:.+]] = arith.cmpi sge, %[[I]], %[[ZERO0]]
      // PERDIM:         %[[UB0:.+]] = arith.cmpi slt, %[[I]], %[[DIM0]]
      // PERDIM:         %[[BOUND0:.+]] = arith.andi %[[LB0]], %[[UB0]]
      // PERDIM:         cf.assert %[[BOUND0]], "memref access out of bounds along dimension 0"
      //
      // PERDIM:         %[[ONE:.+]] = arith.constant 1
      // PERDIM:         %[[DIM1:.+]] = memref.dim %{{.*}}, %[[ONE]]
      // PERDIM:         %[[LB1:.+]] = arith.cmpi sge, %[[J]], %[[ZERO0]]
      // PERDIM:         %[[UB1:.+]] = arith.cmpi slt, %[[J]], %[[DIM1]]
      // PERDIM:         %[[BOUND1:.+]] = arith.andi %[[LB1]], %[[UB1]]
      // PERDIM:         cf.assert %[[BOUND1]], "memref access out of bounds along dimension 1"
      //
      // CHECK:          memref.store
      memref.store %2, %memref[%i, %j] : memref<?x?xf32>
      scf.yield %sum : index
    }
  }
  return
}
