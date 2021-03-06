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
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

using namespace mlir;

namespace mutate{
void collectValueBeforeOp(FuncOp &F, Operation* boundary, Value refV,
                         std::vector<std::pair<Operation*, std::string>> &resultVec);
std::pair<Operation*, std::string> randValueBeforeOp(FuncOp &F, Operation* boundary, Value refV);
bool replaceAllUsesWithReport(Value from, std::pair<Operation*, std::string> &metaTo);
Operation* getLocalConstantOp(Operation *op, Value refV);
std::vector<mlir::Operation*> traverseNestedOp(mlir::Operation* startop, 
                                               bool insertStartOp=false,
                                               bool excludeIsolatedFromAbove=false);
void updateUID(Operation* srcOp, Operation* dstOp, std::string mode);
bool replaceUnfulfillOperands(Operation *op);
void useResult(Operation *op);

Value& TensorReshape(Value& from, const Value& to, std::string fromDesc);

Operation* walkCollect(std::string opDesc, std::string &UID, ModuleOp &m);
Operation* walkExact(std::string opDesc, std::string &UID, ModuleOp &m);
Operation* walkPosition(std::string opDesc, std::string &UID, ModuleOp &m);

/**
 * Insert an NOP (using mlir::ConstantOp) before refOp is removed
 **/
Operation* insertNOP(Operation *refOp);

class Insert : public PassWrapper<Insert, OperationPass<ModuleOp>> {
public:
  Insert() {}
  Insert(std::string dst, std::string src, bool noResult) : 
    dstDesc(dst), srcDesc(src), noResult(noResult){}
  void runOnOperation() override;
  
private:
  std::string dstDesc;
  std::string srcDesc;
  bool noResult;
};

class OperandReplace : public PassWrapper<OperandReplace, OperationPass<ModuleOp>> {
public:
  OperandReplace() {}
  OperandReplace(std::string dst, std::string src) : 
    dstDesc(dst), srcDesc(src){}
  void runOnOperation() override;
  
private:
  std::string dstDesc;
  std::string srcDesc;
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
class List : public PassWrapper<List, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};
}

#endif //MUTATE_H