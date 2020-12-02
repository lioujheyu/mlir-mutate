# Introduction
mlir-mutate aims to manipluate any operation sequence from a given MLIR input and will be used by upcoming Genetic Programming framework like [GEVO](https://github.com/lioujheyu/gevo). It is originally designed to work with [TensorFlow Runtime](https://github.com/tensorflow/runtime) MLIR dialect but quickly switch to [Google IREE](https://github.com/google/iree) with usable frontend-backend compile flow. This tool is developed along with GEVO-ML but can be used as indepedant tool performing MLIR-operation wised copy and deletion with MLIR file generated and compiled from Google IREE (MHLO and flow dialect). 

# Installation
## Prerequisite 
mlir-mutate requries LLVM-project with modified MLIR code base to support unique ID field. This requirement is included as a git submodule to this [llvm-forked repository](https://github.com/lioujheyu/llvm-project)
```shell
  git submodule update --init
```

## Compile
As simple as 
```shell
  mkdir build 
  cd build 
  cmake ../
  make
```
