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

## Developing – Pre-commit

Please use pre-commit by installing it from
[pre-commit.com](https://pre-commit.com) or system repository and running
`pre-commit` in the repository once. After that, every further commit will run
through the pre-commit checks such as formatters and linters. If any problems
are found, please fix them and amend the commit before pushing.

## Developing – Certificate of Origin

Please follow the [Developer Certificate of
Origin](https://wiki.linuxfoundation.org/dco) policy by signing off the
commits, e.g., use `git commit -s` to automatically add the required field.
