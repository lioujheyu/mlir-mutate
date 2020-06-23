#ifndef MUTATE_H
#define MUTATE_H

#include <random>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Types.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"

#include "tfrt/basic_kernel/basic_kernels.h"

using namespace mlir;

namespace mutate{
/***
 * Because the god damn lifetime of OpResult only last in the scope of Operation.getResult()
 * , the implementation here actually uses "operation + result index" in the std::pair instead 
 * of value as the reference. However, since the function argument can also be the desired value,
 * I have to use void pointer to store operation and blockargument passing through the function
 * and later on casting to the the correct type based on the UID string. 
 * */
void collectValueBeforeOp(FuncOp &F, Operation* boundary, Value refV,
                         std::vector<std::pair<void*, std::string>> &resultVec);
std::pair<void*, std::string> randValueBeforeOp(FuncOp &F, Operation* boundary, Value refV);
void replaceAllUsesWithReport(Value from, std::pair<void*, std::string> &metaTo);
std::vector<mlir::Operation*> traverseNestedOp(mlir::Operation* startop, bool insertStartOp=false);
Operation* walkExact(std::string op_desc, std::string &UID, ModuleOp &m);

/**
 * Insert an NOP (using mlir::std::ConstantOp) after refOp
 **/
Operation* insertNOP(Operation *refOp);

class Insert : public PassWrapper<Insert, OperationPass<ModuleOp>> {
public:
  Insert() {}
  Insert(std::string src, std::string dst) : srcOp(src), dstOp(dst){}
  void runOnOperation() override;
  
private:
  std::string srcOp;
  std::string dstOp;
};

class Cut : public PassWrapper<Cut, OperationPass<ModuleOp>> {
public:
  Cut() {}
  Cut(std::string targetop) : op1(targetop) {}
  void runOnOperation() override;
  
private:
  std::string op1;
};

class Name : public PassWrapper<Name, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};
}

#endif //MUTATE_H