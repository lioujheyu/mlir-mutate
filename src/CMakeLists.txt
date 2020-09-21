get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
include(CMakePrintHelpers)
cmake_print_variables(dialect_libs)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMIR
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTargetLLVMIR
    MLIRTransforms
)
add_llvm_executable(mlir-mutate main.cpp mutate.cpp)

llvm_update_compile_flags(mlir-mutate)
target_link_libraries(mlir-mutate PRIVATE ${LIBS})
