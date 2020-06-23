#include "mutate.h"
#include "mlir/IR/Dominance.h"

std::random_device rd;
std::mt19937 gen(rd());

void mutate::collectValueBeforeOp(FuncOp &F, Operation* boundary, Value refV,
                           std::vector<std::pair<void*, std::string>> &resultVec)
{
    for (BlockArgument &A: F.getArguments()) {
        if (refV != nullptr) {
            if (A.getType() != refV.getType())
                continue;
        }
        std::string tmp = "A" + std::to_string(A.getArgNumber());
        resultVec.push_back(std::make_pair((void*)&A, tmp));
    }

    for (Operation &parentop: F.getCallableRegion()->getOps()) {
    std::vector<Operation*> returnops;
    returnops = traverseNestedOp(&parentop, true);
    for (Operation *op : returnops) {
        if (boundary != nullptr) {
            DominanceInfo dom = DominanceInfo(boundary);
            if (op == boundary)
                return;
            if (dom.dominates(op, boundary) == false)
                continue;
        }
        if (op->getUID().find("nop") != std::string::npos)
            continue;

        for (OpResult v : op->getResults()) {
            if (v == refV)
                continue;
            // if (v.getType()->isVoidTy())
            //     continue;
            if (refV != nullptr) {
                if (v.getType() != refV.getType())
                    continue;
                // if (T->isPointerTy()) {
                //     if (I.getType()->getPointerElementType() !=
                //         T->getPointerElementType())
                //         continue;
                // }
            }
            std::string UID = std::to_string(v.getResultNumber());
            resultVec.push_back(std::make_pair((void*)op, UID));
        }
    }
    }
}

std::pair<void*, std::string> mutate::randValueBeforeOp(FuncOp &F, Operation* boundary, Value refV)
{
    std::vector<std::pair<void*, std::string>> resultVec;
    collectValueBeforeOp(F, boundary, refV, resultVec);
    // has constant to participate in drawing
    // resultVec.push_back(std::make_pair(getConstantValue(refOP->getType()), StringRef("C1")));
    if (resultVec.empty())
        return std::make_pair(nullptr, "");

    std::uniform_int_distribution<> randIdx(0, resultVec.size()-1);
    return resultVec[randIdx(gen)];
}

void mutate::replaceAllUsesWithReport(Value from, std::pair<void*, std::string> &metaTo)
{
    std::vector<OpOperand*> useList;
    for(OpOperand &use : from.getUses())
        useList.push_back(&use);

    while(!useList.empty()) {
        OpOperand *use = useList.back();
        Operation *useOp = use->getOwner();
        std::string FromUID = useOp->getUID() + ".OP" + std::to_string(use->getOperandNumber());
        llvm::errs()<<"opreplaced "<< FromUID;
        if (metaTo.second[0] == 'A') {
            use->set(*((BlockArgument*)metaTo.first));
            llvm::errs() << "," << metaTo.second << "\n";
        }
        else {
            int resultIdx = std::stoi(metaTo.second);
            Operation *toOp = (Operation*)metaTo.first;
            use->set(toOp->getResult(resultIdx));
            llvm::errs() << "," << toOp->getUID() + "_" + metaTo.second << "\n";
        }
        useList.pop_back();
    }
}

std::vector<Operation*> mutate::traverseNestedOp(Operation* startop, bool insertStartOp) {
    std::vector<Operation*> ops;

    if (insertStartOp)
        ops.push_back(startop);
  
    if (startop->getNumRegions() == 0)
        return ops;

    for (int i=0; i<startop->getNumRegions(); ++i){
        for (auto &op: startop->getRegion(i).getOps()) {
            ops.push_back(&op);
            std::vector<Operation*> nestedops = traverseNestedOp(&op);
            if (!nestedops.empty())
                ops.insert(ops.end(), nestedops.begin(), nestedops.end());
        }
    }
    return ops;
}

Operation* mutate::walkExact(std::string op_desc, std::string &UID, ModuleOp &m) {
    for (FuncOp f : m.getOps<FuncOp>()) {
    for (auto &parentop: f.getCallableRegion()->getOps()) {
        std::vector<Operation*> returnops;
        returnops = traverseNestedOp(&parentop, true);
        for (Operation *op: returnops) {
            if (op_desc == op->getUID())
                return op;
        }
    }
    } 

    return nullptr;
}

Operation* mutate::insertNOP(Operation *refOp) {
    assert(refOp->getParentRegion());

    OpBuilder builder(refOp->getContext());
    builder.setInsertionPoint(refOp);
    auto nop = builder.create<mlir::ConstantIntOp>(builder.getUnknownLoc(), 0, 32);
    nop.getOperation()->setUID(refOp->getUID() + ".d");
    return nop.getOperation();
}

void mutate::Cut::runOnOperation() {
    // Get the current operation being operated on.
    ModuleOp module = getOperation();
    std::string rUID;
    Operation* op = walkExact(op1, rUID, module);
    if (op == nullptr) {
        llvm::errs() << "cut failed. Cannot find " << op1 << "\n";
        return;
    }

    FuncOp f = op->getParentOfType<FuncOp>();
    if (!op->use_empty()) {
        for (OpResult v : op->getResults()) {
            std::pair<void*, std::string> metaV =
                randValueBeforeOp(f, op, v);
            
            if(metaV.first == nullptr) {
                llvm::errs() << "cut failed. Cannot find result replacement for " <<
                    op1+"_"+std::to_string(v.getResultNumber()) << "\n";
                return;
            }
                
            replaceAllUsesWithReport(v, metaV);
        }
    }
    insertNOP(op);
    op->erase();
    llvm::errs() << "cut " << op1 << "\n";
}

void mutate::Insert::runOnOperation() {
    // Get the current operation being operated on.
    ModuleOp module = getOperation();
}

void mutate::Name::runOnOperation() {
    ModuleOp module = getOperation();
    std::vector<Operation*> ops;
    for (FuncOp f : module.getOps<FuncOp>()) {
        for (auto &op: f.getCallableRegion()->getOps()) {
            ops.push_back(&op);
            std::vector<Operation*> returnops;
            returnops = traverseNestedOp(&op);
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
