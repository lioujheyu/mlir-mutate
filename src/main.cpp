#include <stdio.h>
#include <vector>

#include "mlir/Pass/PassManager.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mutate.h"

using namespace mlir;

std::unique_ptr<Pass> createCutPass(std::string op1) {
  return std::make_unique<mutate::Cut>(op1);
}
std::unique_ptr<Pass> createNamePass() {
  return std::make_unique<mutate::Name>();
}
static PassRegistration<mutate::Cut> cutpass(
    "cut", "delete an operation");
static PassRegistration<mutate::Name> namepass(
    "name", "delete an operation");

int main(int argc, char **argv) {
  llvm::cl::OptionCategory MlirMutateOptions(
    "MLIR Mutation Options",
    "These are options for mlir-mutate."
  );
  llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<MLIR Input file>"),
    llvm::cl::init("-"),
    llvm::cl::cat(MlirMutateOptions)
  );
  llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output filename"),
    llvm::cl::value_desc("filename"), 
    llvm::cl::init(""),
    llvm::cl::cat(MlirMutateOptions)
  );
  llvm::cl::opt<std::string> cutOp(
    "c", llvm::cl::desc("delete an operation"),
    llvm::cl::cat(MlirMutateOptions)
  );
  llvm::cl::opt<bool> nameOp(
    "n", llvm::cl::desc("name all operation with unique ID"),
    llvm::cl::cat(MlirMutateOptions)
  );

  llvm::cl::HideUnrelatedOptions(MlirMutateOptions);
  llvm::cl::ParseCommandLineOptions(
    argc, argv,
    "MLIR mutate operations\n"
  );
  mlir::DialectRegistration<tfrt::hex::HexDialect>();
  mlir::registerAllDialects();

  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  mlir::OwningModuleRef module;
  mlir::PassManager pm(&context, false);

  // Handle '.mlir' input to the ONNX MLIR frontend.
  // The mlir format indicates that one or more of the supported
  // representations are used in the file.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return -1;
  }

  if (nameOp)
    pm.addPass(createNamePass());
  else {
    if (!cutOp.empty())
      pm.addPass(createCutPass(cutOp));
  }

  pm.run(module.get());

  if (!outputFilename.empty()) {
    freopen(outputFilename.c_str(), "w", stderr);
    module.get().dump();
    fclose(stderr);
  }
  else
    module.get().dump();
  
  return 0;
}