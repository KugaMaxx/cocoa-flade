#!/bin/bash

ROOT=$(cd "$(dirname "$0")";pwd)

# Build Generator
cd ${ROOT}/generator
rm -r build
mkdir build
cd build
cmake ..
make

# Build Extractor
cd ${ROOT}/extractor
rm -r build
mkdir build
cd build
cmake ..
make
