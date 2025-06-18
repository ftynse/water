// RUN: water-opt %s --water-hello-world | FileCheck %s

// CHECK: module
// expected-remark @below {{Hello, world!}}
builtin.module {}
