// RUN: not water-opt %s -diagnostics-file - | FileCheck %s

// File with intentionally broken syntax to see that we capture errors and
// print them into a JSON file.

// CHECK: {"column":9,"file":"{{.*}}json-diagnostics.mlir","line":7,"message":"expected operation name in quotes","severity":"error"}
module {
