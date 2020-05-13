add_mlir_dialect_library(hex
        basic_kernels.cc
#        static_registration.cc

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/tfrt/basic_kernel
        DEPENDS
        MLIRbasic_kernelsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)