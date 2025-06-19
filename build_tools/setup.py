# Copyright 2025 The Water Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from setuptools import setup, Extension
import subprocess
import os
import shutil
import sys
from distutils.command.build import build as _build
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.egg_info import egg_info
from pathlib import Path

SETUPPY_DIR = os.path.realpath(os.path.dirname(__file__))
BINARY_DIR = os.path.join(SETUPPY_DIR, "build")


def prepare_installation():
    subprocess.check_call(["cmake", "--version"])
    cmake_args = [
        "-G Ninja",
        "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
        # TODO: this should be found somehow...
        "-DMLIR_DIR=/home/azinenko/git/llvm-project/build/lib/cmake/mlir",
        "-DBUILD_SHARED_LIBS=Off",
    ]
    source_dir = os.path.dirname(SETUPPY_DIR)
    os.makedirs(BINARY_DIR, exist_ok=True)
    subprocess.check_call(["cmake", source_dir, *cmake_args], cwd=BINARY_DIR)
    subprocess.check_call(
        ["cmake", "--build", ".", "--target", "water-opt"], cwd=BINARY_DIR
    )


class CMakeBuildPy(_build_py):
    def run(self):
        target_dir = os.path.abspath(self.build_lib)
        print(f"Building in target dir: {target_dir}", file=sys.stderr)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir)
        shutil.copy(os.path.join(BINARY_DIR, "bin", "water-opt"), target_dir)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class NoopBuildExtension(_build_ext):
    def __init__(self, *args, **kwargs):
        assert False

    def build_extension(self, ext):
        pass


class CustomBuild(_build):
    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")
        self.run_command("build_scripts")


# I don't know. Something about the CMake 'install' is producing a .egg-info/
# folder, which then get picked up by the .whl. For release wheels all we need
# is a .dist-info/ folder, so delete the .egg-info/ folder.
#
# * Notes: https://github.com/iree-org/iree/issues/19155
# * Implementation inspirted by https://stackoverflow.com/a/70146326
class CleanEggInfo(egg_info):
    def run(self):
        print(f"CleanEggInfo checking: '{BINARY_DIR}'")
        for d in Path(BINARY_DIR).glob("*.egg-info"):
            print(f"found egg-info path '{d}', deleting")
            shutil.rmtree(d, ignore_errors=True)

        egg_info.run(self)


prepare_installation()

setup(
    name="water_mlir",
    version="0.1.0",
    ext_modules=[CMakeExtension("water_mlir")],
    cmdclass={
        "build": CustomBuild,
        "built_ext": NoopBuildExtension,
        "build_py": CMakeBuildPy,
        "egg_info": CleanEggInfo,
    },
)
