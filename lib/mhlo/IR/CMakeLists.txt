set(LLVM_TARGET_DEFINITIONS hlo_patterns.td)
mlir_tablegen(hlo_patterns.cc.inc -gen-rewriters)
add_public_tablegen_target(MLIRhlo_patternsIncGen)

set(LLVM_TARGET_DEFINITIONS mhlo_canonicalize.td)
mlir_tablegen(mhlo_canonicalize.inc -gen-rewriters)
add_public_tablegen_target(MLIRmhlo_canonicalizeIncGen)

add_mlir_dialect_library(hlo
        hlo_ops.cc
        hlo_ops_base_structs.cc
        hlo_utils.cc
        convert_op_folder.cc
        # "hlo_ops.cc.inc"
        # "hlo_ops.h.inc"
        # chlo_ops.cc
        # infer_fusibility_op_interface.cc

        # ADDITIONAL_HEADER_DIRS
        # ${PROJECT_SOURCE_DIR}/include/tfrt/basic_kernel
        DEPENDS
        MLIRhlo_opsIncGen
        MLIRhlo_ops_baseIncGen
        MLIRinfer_fusibility_op_interfaceIncGen
        MLIRhlo_patternsIncGen
        MLIRmhlo_canonicalizeIncGen
        LLVMSupport
        MLIRIR
        MLIRSideEffectInterfaces
        MLIRStandard
        MLIRSupport
        MLIRTransformUtils

	LINK_LIBS PUBLIC
	MLIRIR
	)