#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Function.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "tfrt/basic_kernel/basic_kernels.h"

using namespace mlir;

namespace{
class Cut : public PassWrapper<Cut, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Get the current operation being operated on.
    ModuleOp module = getOperation();
    for (FuncOp f : module.getOps<FuncOp>()) {
      llvm::errs() << "Function: " << f.getName() << "\n";
      for (auto &op: f.getCallableRegion()->getOps()) {
        op.walk([](Operation *nestedOp) {
          llvm::errs() << nestedOp->getName() << ":" << nestedOp->getLoc() << "\n";
        });
        if (op.use_empty()) {
          op.erase();
          break;
        }
      }
    }
  }
};

class Name : public PassWrapper<Name, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    int cnt = 0;
    std::string uid;
    for (FuncOp f : module.getOps<FuncOp>()) {
      for (auto &op: f.getCallableRegion()->getOps()) {
        uid = "U" + std::to_string(cnt);
        cnt++;
        llvm::errs() << op.getName() << "\n";
        op.setAttr("UID", StringAttr::get(uid, op.getContext()));
      }
    }
  }
};

}

std::unique_ptr<Pass> createCutPass() {
  return std::make_unique<Cut>();
}
std::unique_ptr<Pass> createNamePass() {
  return std::make_unique<Name>();
}
static PassRegistration<Cut> cutpass(
    "cut", "delete an operation");
static PassRegistration<Name> namepass(
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
  // llvm::cl::opt<std::string> outputFilename(
  //   "o", llvm::cl::desc("Output filename"),
  //   llvm::cl::value_desc("filename"), 
  //   llvm::cl::init("-"),
  //   llvm::cl::cat(MlirMutateOptions)
  // );
  llvm::cl::opt<std::string> cutOp(
    "c", llvm::cl::desc("delete an operation"),
    llvm::cl::cat(MlirMutateOptions)
  );
  llvm::cl::opt<std::string> nameOp(
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
  mlir::PassManager pm(&context);

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

  if (!nameOp.empty()) {
    pm.addPass(createNamePass());
  }
  else if (!cutOp.empty()) {
    pm.addPass(createCutPass());
  }

  pm.run(module.get());

  module.get().dump();

  return 0;
}