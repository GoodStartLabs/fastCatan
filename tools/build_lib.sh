#!/usr/bin/env bash
# Build the ctypes shared lib for Python tests.
# Output: build/libfastcatan.{dylib,so}
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p build

case "$(uname -s)" in
    Darwin) EXT=dylib ;;
    Linux)  EXT=so ;;
    *)      echo "unsupported OS: $(uname -s)"; exit 1 ;;
esac

clang++ -std=c++23 -O2 -fPIC -shared -Wall -Wextra \
    -I include \
    src/catan/rules.cpp tools/c_api.cpp \
    -o "build/libfastcatan.${EXT}"

echo "built build/libfastcatan.${EXT}"
