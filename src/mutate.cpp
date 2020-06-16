#include "mutate.h"

std::vector<mlir::Operation*> mutate::traverseNestedOp(mlir::Operation* startop) {
  std::vector<mlir::Operation*> ops;
  
  if (startop->getNumRegions() == 0)
    return ops;

  for (int i=0; i<startop->getNumRegions(); ++i){
    for (auto &op: startop->getRegion(i).getOps()) {
      ops.push_back(&op);
      std::vector<mlir::Operation*> nestedops = traverseNestedOp(&op);
      if (!nestedops.empty())
        ops.insert(ops.end(), nestedops.begin(), nestedops.end());
    }
  }
  return ops;
}

void mutate::Cut::runOnOperation() {
    // Get the current operation being operated on.
    ModuleOp module = getOperation();
    for (FuncOp f : module.getOps<FuncOp>()) {
      for (auto &parentop: f.getCallableRegion()->getOps()) {
        std::vector<Operation*> returnops;
        returnops = mutate::traverseNestedOp(&parentop);
        returnops.insert(returnops.begin(), &parentop);
        for (auto *op: returnops) {
          if (op1 == op->getUID()) {
            returnops = mutate::traverseNestedOp(op);
            for (auto *nestop: returnops)
              nestop->erase();
            op->erase();
            return;
          }
        }
      }
    }
    llvm::errs() << op1 << " not found" << '\n' ;
}

void mutate::Name::runOnOperation() {
    ModuleOp module = getOperation();
    std::vector<Operation*> ops;
    for (FuncOp f : module.getOps<FuncOp>()) {
      for (auto &op: f.getCallableRegion()->getOps()) {
        ops.push_back(&op);
        std::vector<Operation*> returnops;
        returnops = mutate::traverseNestedOp(&op);
        ops.insert(ops.end(), returnops.begin(), returnops.end());
      }
    }

    int cnt = 0;
    std::string uid;
    for (auto *op: ops) {
      uid = "U" + std::to_string(cnt);
      cnt++;
      // op->setAttr("UID", StringAttr::get(uid, op->getContext()));
      op->setUID(uid);
    }
}
