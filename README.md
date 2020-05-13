# Introduction
mlir-mutate aims to manipluate any operation sequence from a given MLIR input and will be used by upcoming Genetic Programming framework like [GEVO](https://github.com/lioujheyu/gevo). It's current design targeting [TensorFlow Runtime](https://github.com/tensorflow/runtime) MLIR dialect as this project seems to be the only driver able to execute MLIR with proper modular capablility. mlir-mutate is still under heavy construction and currently it can only parse the MLIR file from TensorFlow runtime testing repository.

# Installation
## Prerequisite 
* [LLVM-project with MLIR](https://mlir.llvm.org/getting_started/)

  After compiling llvm-project following the above link, please install the LLVM and MLIR through `"ninja install"`. mlir-mutate depends on the MLIR cmake files that available in the system path (by default `/usr/local/`) so that it can properly locate the necessary MLIR libraries. 

## Compile
As simple as 
```shell
  mkdir build 
  cd build 
  cmake ../
  make
```
