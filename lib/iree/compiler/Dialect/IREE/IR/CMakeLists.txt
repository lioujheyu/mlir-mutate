set(LLVM_TARGET_DEFINITIONS IREEOps.td)
mlir_tablegen(IREEOps.cpp.inc -gen-op-defs -I ${PROJECT_SOURCE_DIR}/lib)
mlir_tablegen(IREEOps.h.inc -gen-op-decls -I ${PROJECT_SOURCE_DIR}/lib)
add_public_tablegen_target(MLIRIREEOpsIncGen)

add_mlir_dialect_library(IREEIR
    IREEDialect.cpp
    IREEOps.cpp
    IREEOps.cpp.inc
    IREETypes.cpp

    # ADDITIONAL_HEADER_DIRS
    # ${PROJECT_BINARY_DIR}/lib

    DEPENDS
    MLIRIREEOpsIncGen
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRStandardOps
    MLIRSupport

    LINK_LIBS PUBLIC
    MLIRIR
)

# iree_cc_library(
#   NAME
#     IR
#   HDRS
#     "IREEAttributes.h"
#     "IREEDialect.h"
#     "IREEOps.h"
#     "IREEOps.h.inc"
#     "IREETraits.h"
#     "IREETypes.h"
#   SRCS
#     "IREEDialect.cpp"
#     "IREEOps.cpp"
#     "IREEOps.cpp.inc"
#     "IREETypes.cpp"
#   DEPS
#     LLVMSupport
#     MLIRIR
#     MLIRParser
#     MLIRSideEffectInterfaces
#     MLIRStandardOps
#     MLIRSupport
#   PUBLIC
# )