# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The ML SDK Model Converter is a C++17 command-line application that translates TOSA (Tensor Operator Set Architecture) ML models to VGF (Vulkan GPU Format) files for execution through Vulkan ML extensions. It's part of ARM's ML SDK for Vulkan.

## Build Commands

### Building the project
```bash
# Standard build from repo manifest checkout (uses default paths)
python3 scripts/build.py -j $(nproc)

# Build with tests and linting enabled
python3 scripts/build.py -j $(nproc) --test --lint

# Build with documentation
python3 scripts/build.py -j $(nproc) --doc

# Build with custom dependency paths
python3 scripts/build.py -j $(nproc) \
    --vgf-lib-path ${PATH_TO_VGF_LIB_CHECKOUT} \
    --flatbuffers-path ${PATH_TO_FLATBUFFERS_CHECKOUT} \
    --argparse-path ${PATH_TO_ARGPARSE_CHECKOUT} \
    --tosa-mlir-translator-path ${PATH_TO_TOSA_MLIR_TRANSLATOR_CHECKOUT} \
    --external-llvm ${PATH_TO_LLVM_CHECKOUT}
```

### Running tests
```bash
# Run tests after build
python3 scripts/build.py -j $(nproc) --test

# Run tests manually with pytest
python -m pytest test -n $(nproc) --build-dir build --build-type Release

# Run specific test file
python -m pytest test/test_model_converter_v100.py --build-dir build --build-type Release
```

### Linting and code quality
```bash
# C++ linting with cppcheck (enabled during build)
python3 scripts/build.py -j $(nproc) --lint

# Python formatting with Black
black .

# Run all pre-commit hooks
pre-commit run --all-files --hook-stage commit
pre-commit run --all-files --hook-stage push

# Run specific pre-commit hook
pre-commit run clang-format
```

## Architecture

### Core Components
- **Compiler** (`src/compiler.cpp/hpp`) - Main entry point for model conversion
- **Conversion Passes** (`src/conversion/`) - MLIR transformation passes for TOSA to VGF conversion
- **VGF Dialect** (`src/vgf-dialect/`) - MLIR dialect implementation for VGF operations
- **Optimizer** (`src/opt.cpp`) - MLIR optimization tool for testing passes

### Key Technologies
- **MLIR/LLVM** - Infrastructure for model transformations
- **Flatbuffers** - Serialization for TOSA models
- **SPIR-V** - Generated modules for Vulkan GPU execution
- **CMake** - Build system with sophisticated dependency management

### Input/Output Formats
- **Inputs**: TOSA FlatBuffers, TOSA MLIR bytecode, TOSA MLIR textual format
- **Outputs**: VGF files (containing SPIR-V modules), TOSA FlatBuffers

## Development Workflow

1. **Code Style**: Follow CppCoreGuidelines for C++, use clang-format for automatic formatting
2. **Pre-commit Hooks**: Install with `pre-commit install` to ensure code quality
3. **DCO Sign-off**: All commits must include `Signed-off-by: Your Name <email>` 
4. **Testing**: Write tests in `/test/` directory using pytest framework
5. **Documentation**: Update RST docs in `/docs/source/` when adding features

## Important Notes

- The build script (`scripts/build.py`) is the primary interface for building and testing
- When working from a repo manifest checkout, dependency paths are automatic
- Tests require `--build-dir` and `--build-type` parameters when run manually
- Use `model-converter --help` to see runtime options
- LLVM patches may be applied during build (can skip with `--skip-llvm-patch`)
- Support for Linux (GCC/Clang) and Windows (MSVC) host platforms