#include <stdio.h>
#include <iostream>
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
std::unique_ptr<Pass> createOperandReplacePass(std::string dst, std::string src) {
  return std::make_unique<mutate::OperandReplace>(dst, src);
}
std::unique_ptr<Pass> createInsertPass(std::string dst, std::string src, bool noResult) {
  return std::make_unique<mutate::Insert>(dst, src, noResult);
}
std::unique_ptr<Pass> createNamePass() {
  return std::make_unique<mutate::Name>();
}
std::unique_ptr<Pass> createListPass() {
  return std::make_unique<mutate::List>();
}
static PassRegistration<mutate::Cut> cutpass(
    "cut", "delete an operation");
static PassRegistration<mutate::OperandReplace> operandreplacepass(
    "operandreplace", "Replace an operand with a result");
static PassRegistration<mutate::Insert> insertpass(
    "insert", "Copy an operation to somewhere else");
static PassRegistration<mutate::Name> namepass(
    "name", "Name each operation with a unique ID");
static PassRegistration<mutate::List> listpass(
    "list", "Count number of operation in the program");

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
    "c", llvm::cl::desc("Delete an operation"),
    llvm::cl::value_desc("OpUID"),
    llvm::cl::cat(MlirMutateOptions)
  );
  llvm::cl::list<std::string> OperandReplaceOp(
    "p", llvm::cl::desc("Replace an operand with a result"),
    llvm::cl::value_desc("dstOperandUID,srcResultUID"),
    llvm::cl::cat(MlirMutateOptions),
    llvm::cl::CommaSeparated
  );
  llvm::cl::list<std::string> InsertOp(
    "i", llvm::cl::desc("Copy an operation to somewhere else"),
    llvm::cl::value_desc("dstOpUID,srcOpUID"),
    llvm::cl::cat(MlirMutateOptions),
    llvm::cl::CommaSeparated
  );
  llvm::cl::opt<bool> nameOp(
    "n", llvm::cl::desc("Name each operation with a unique ID"),
    llvm::cl::cat(MlirMutateOptions)
  );

  llvm::cl::opt<bool> listOp(
    "l", llvm::cl::desc("Count number of operation in the program"),
    llvm::cl::cat(MlirMutateOptions)
  );

  llvm::cl::opt<bool> noUID(
    "nouid", llvm::cl::desc("No unique ID for MLIR program output"),
    llvm::cl::cat(MlirMutateOptions)
  );

  llvm::cl::opt<bool> noResult(
    "not_use_result", llvm::cl::desc("No unique ID for MLIR program output"),
    llvm::cl::init(false),
    llvm::cl::cat(MlirMutateOptions)
  );

  llvm::cl::HideUnrelatedOptions(MlirMutateOptions);
  llvm::cl::ParseCommandLineOptions(
    argc, argv,
    "MLIR mutate operations\n"
  );

  mlir::registerAllDialects();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  

  mlir::MLIRContext context;
  context.getOrLoadDialect<iree_compiler::IREE::Flow::FlowDialect>();
  context.getOrLoadDialect<iree_compiler::IREEDialect>();
  context.getOrLoadDialect<mhlo::MhloDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  mlir::OwningModuleRef module;
  mlir::PassManager pm(&context, true);

  if (!inputFilename.empty()) {
    // Handle '.mlir' input 
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
  }
  else {
    std::string lineInput;
    while( std::cin >> lineInput ) {}
    module = parseSourceString(lineInput, &context);
    if (!module)
      return -1;
  }

  if (nameOp)
    pm.addPass(createNamePass());
  else if (listOp)
    pm.addPass(createListPass());
  else {
    if (!cutOp.empty())
      pm.addPass(createCutPass(cutOp));
    if (!OperandReplaceOp.empty()) {
      if (OperandReplaceOp.size() != 2) {
        llvm::errs() << "Insertion operation needs 2 arguments\n";
        return -1;
      }
      pm.addPass(createOperandReplacePass(OperandReplaceOp[0], OperandReplaceOp[1]));
    }
    if (!InsertOp.empty()) {
      if (InsertOp.size() != 2) {
        llvm::errs() << "Insertion operation needs 2 arguments\n";
        return -1;
      }
      pm.addPass(createInsertPass(InsertOp[0], InsertOp[1], noResult));
    }
  }

  if (failed(pm.run(*module)))
    return -1;

  if (listOp)
    return 0;
  
  if (!outputFilename.empty()) {
    freopen(outputFilename.c_str(), "w", stderr);
    if (noUID)
      module.get().print(llvm::errs(), OpPrintingFlags());
    else
      module.get().print(llvm::errs(), OpPrintingFlags().enableUID());
    fclose(stderr);
  }
  else {
    if (noUID)
      module.get().print(llvm::outs(), OpPrintingFlags());
    else
      module.get().print(llvm::outs(), OpPrintingFlags().enableUID());
  }
  
  return 0;
}
