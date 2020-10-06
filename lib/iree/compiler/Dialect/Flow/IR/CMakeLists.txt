set(LLVM_TARGET_DEFINITIONS FlowBase.td)
mlir_tablegen(FlowEnums.cpp.inc -gen-op-defs -I ${PROJECT_SOURCE_DIR}/lib)
mlir_tablegen(FlowEnums.h.inc -gen-op-decls -I ${PROJECT_SOURCE_DIR}/lib)
mlir_tablegen(FlowOpInterface.cpp.inc -gen-op-interface-defs -I ${PROJECT_SOURCE_DIR}/lib)
mlir_tablegen(FlowOpInterface.h.inc -gen-op-interface-decls -I ${PROJECT_SOURCE_DIR}/lib)
add_public_tablegen_target(MLIRFlowBaseIncGen)

# set(LLVM_TARGET_DEFINITIONS FlowBase.td)

# add_public_tablegen_target(MLIRFlowOpInterfaceGenIncGen)

set(LLVM_TARGET_DEFINITIONS FlowOps.td)
mlir_tablegen(FlowOps.cpp.inc -gen-op-defs -I ${PROJECT_SOURCE_DIR}/lib)
mlir_tablegen(FlowOps.h.inc -gen-op-decls -I ${PROJECT_SOURCE_DIR}/lib)
add_public_tablegen_target(MLIRFlowOpsGenIncGen)

add_mlir_dialect_library(FlowIR
    "FlowDialect.cpp"
    "FlowEnums.cpp.inc"
    "FlowOpFolders.cpp"
    "FlowOpInterface.cpp.inc"
    "FlowOpUtils.cpp"
    "FlowOps.cpp"
    "FlowOps.cpp.inc"
    "FlowTypes.cpp"

    # ADDITIONAL_HEADER_DIRS
    # ${PROJECT_SOURCE_DIR}/lib/iree/compiler/Dialect/Flow/IR
    # ${PROJECT_SOURCE_DIR}/lib/iree/compiler/Dialect/IREE/IR

    DEPENDS
    MLIRIREEOpsIncGen
    MLIRFlowBaseIncGen
    LLVMSupport
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRStandard
    MLIRSupport
    MLIRTransformUtils

	LINK_LIBS PUBLIC
    MLIRIR
    IREEIR
	)
