#add_mlir_dialect(basic_kernels hex)
# add_mlir_doc(basic_kernels -gen-dialect-doc Hex_Dialect basic_kernel/)
#add_mlir_doc(basic_kernels -gen-op-doc Hex_Op basic_kernel/)

set(LLVM_TARGET_DEFINITIONS basic_kernels.td)
mlir_tablegen(basic_kernels_opdefs.cpp.inc -gen-op-defs)
mlir_tablegen(basic_kernels.h.inc -gen-op-decls)
add_public_tablegen_target(MLIRbasic_kernelsIncGen)