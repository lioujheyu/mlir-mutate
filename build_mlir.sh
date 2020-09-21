mkdir build_llvm_project
cd build_llvm_project
cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON 
#  -DCMAKE_BUILD_TYPE=Release \
#  -DCMAKE_C_COMPILER=clang \
#  -DCMAKE_CXX_COMPILER=clang++ \
#  -DLLVM_ENABLE_LLD=ON \

cmake --build . --target check-mlir
