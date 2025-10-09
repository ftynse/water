// RUN: water-opt %s | FileCheck %s

// CHECK: #wave.index_symbol<"A">
func.func private @attr() attributes { test.foo = #wave.index_symbol<"A"> }

// CHECK: #wave.index_symbol<"$T0">
func.func private @attr() attributes { test.foo = #wave.index_symbol<"$0"> }
