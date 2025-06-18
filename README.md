# Water 💧

Water forms waves 🌊, and hosts sharks 🦈 and other eerie creatures 👻.

Water is a collection of [MLIR](https://mlir.llvm.org) components including
passes, dialects and interfaces usable in other projects.

## Building - Component Build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR`.

```sh
cmake -G Ninja \
      -B build \
      -DMLIR_DIR=$BUILD_DIR/lib/cmake/mlir
cmake --build build --target check-water
```

## Building - Monolithic Build

This setup assumes that you build the project as part of a monolithic LLVM
build via the `LLVM_EXTERNAL_PROJECTS` mechanism.  To build LLVM, MLIR, the
example and launch the tests run
```sh
cmake -G Ninja \
      -B build \
      $LLVM_SRC_DIR/llvm \
      -DLLVM_TARGETS_TO_BUILD=host \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_EXTERNAL_PROJECTS=water \
      -DLLVM_EXTERNAL_WATER_SOURCE_DIR=$PWD
cmake --build build --target check-water
```
Here, `$LLVM_SRC_DIR` needs to point to the root of the monorepo.
