#ifndef MUTATE_H
#define MUTATE_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Function.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"

#include "tfrt/basic_kernel/basic_kernels.h"

using namespace mlir;

namespace mutate{
std::vector<mlir::Operation*> traverseNestedOp(mlir::Operation* startop);

class Cut : public PassWrapper<Cut, OperationPass<ModuleOp>> {
public:
  Cut() {}
  Cut(std::string targetop) : op1(targetop){}
  void runOnOperation() override;
  
private:
  std::string op1;
};

class Name : public PassWrapper<Name, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};
}

#endif //MUTATE_H