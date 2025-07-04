# Copyright 2025 The Water Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from setuptools import setup, Extension, find_packages
import subprocess
import os
import shutil
from setuptools.command.build_ext import build_ext
from pathlib import Path


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        sourcedir = os.fspath(Path(sourcedir).resolve())
        self.sourcedir = os.path.dirname(os.path.dirname(sourcedir))
        print(f"sourcedir: {self.sourcedir}")


def invoke_cmake(*args, cwd=None):
    subprocess.check_call(["cmake", *args], cwd=cwd)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        # Ensure CMake is available
        try:
            invoke_cmake("--version")
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        mlir_dir = os.environ.get("WATER_MLIR_DIR", None)
        if not mlir_dir:
            raise RuntimeError(
                "Expected MLIR directory to be provided via the WATER_MLIR_DIR environment variable"
            )
        if not os.path.isdir(mlir_dir) or not os.path.exists(
            os.path.join(mlir_dir, "MLIRConfig.cmake")
        ):
            raise RuntimeError(
                f"WATER_MLIR_DIR={mlir_dir} does not point to the MLIR cmake configuration"
            )

        # Create build directory
        build_dir = os.path.abspath(os.path.join(self.build_temp, ext.name))
        shutil.rmtree(build_dir, ignore_errors=True)
        os.makedirs(build_dir, exist_ok=True)

        # Get extension directory
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve() / "water_mlir"

        print(f"extdir: {extdir}")
        # breakpoint()

        # Configure CMake
        cmake_args = [
            "-G Ninja",
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
            f"-DMLIR_DIR={mlir_dir}",
            "-DBUILD_SHARED_LIBS=OFF",
            f"-DCMAKE_INSTALL_PREFIX={extdir}{os.sep}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
        ]
        invoke_cmake(ext.sourcedir, *cmake_args, cwd=build_dir)

        if not self.dry_run:
            # Build CMake project
            invoke_cmake("--build", ".", "--target", "water-opt/install", cwd=build_dir)


setup(
    name="water_mlir",
    version="0.1.0",
    packages=find_packages(include=["water_mlir"]),
    ext_modules=[CMakeExtension("water_mlir")],
    cmdclass={"build_ext": CMakeBuild},
)
