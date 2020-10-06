#include "mutate.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"

std::random_device rd;
std::mt19937 gen(rd());

void mutate::collectValueBeforeOp(FuncOp &f, Operation* boundary, Value refV,
                                std::vector<std::pair<Operation*, std::string>> &resultVec) {
    Operation *upbound = boundary->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
    if (upbound->getNumRegions() > 0) {
        for (auto &a: upbound->getRegion(0).front().getArguments()) {
            if (refV != nullptr) {
                if (a.getType() != refV.getType()) {
                    if (!a.getType().isa<RankedTensorType>() || 
                        !refV.getType().isa<RankedTensorType>())
                        continue;
                    if (!a.getType().cast<RankedTensorType>().hasStaticShape() ||
                        !refV.getType().cast<RankedTensorType>().hasStaticShape())
                        continue;
                }
            }
            std::string tmp = "A" + std::to_string(a.getArgNumber());
            resultVec.push_back(std::make_pair(upbound, tmp));
        }
    }

    DominanceInfo dom = DominanceInfo(upbound);
    for (Operation *op : traverseNestedOp(upbound)) {
        if (op->getUID()[0] == 'R')
            continue;
        if (boundary != nullptr) {
            if (op == boundary)
                return;
        }
        if (op->getUID().find("nop") != std::string::npos)
            continue;

        for (OpResult v : op->getResults()) {
            if (boundary != nullptr) {
                if (!dom.dominates(v, boundary))
                    continue;
            }

            if (v == refV)
                continue;
            if (refV != nullptr) {
                if (v.getType() != refV.getType()) {
                    if (!v.getType().isa<RankedTensorType>() || 
                        !refV.getType().isa<RankedTensorType>())
                        continue;
                    if (!v.getType().cast<RankedTensorType>().hasStaticShape() ||
                        !refV.getType().cast<RankedTensorType>().hasStaticShape())
                        continue;
                }
            }
            std::string UID = std::to_string(v.getResultNumber());
            resultVec.push_back(std::make_pair(op, UID));
        }
    }
}

std::pair<Operation*, std::string> mutate::randValueBeforeOp(FuncOp &F, Operation* boundary, Value refV) {
    std::vector<std::pair<Operation*, std::string>> resultVec;
    Operation *cop = getLocalConstantOp(boundary, refV);
    if (cop != nullptr)
        resultVec.push_back(std::make_pair(cop, "0"));
    collectValueBeforeOp(F, boundary, refV, resultVec);
    // has constant to participate in drawing
    // resultVec.push_back(std::make_pair(getConstantValue(refOP->getType()), StringRef("C1")));
    if (resultVec.empty())
        return std::make_pair(nullptr, "");

    std::uniform_int_distribution<> randIdx(0, resultVec.size()-1);
    return resultVec[randIdx(gen)];
}

bool mutate::replaceAllUsesWithReport(Value from, std::pair<Operation*, std::string> &metaTo) {
    std::vector<OpOperand*> useList;
    for(OpOperand &use : from.getUses())
        useList.push_back(&use);

    while(!useList.empty()) {
        OpOperand *use = useList.back();
        Operation *useOp = use->getOwner();
        std::string FromUID = useOp->getUID() + "_OP" + std::to_string(use->getOperandNumber());
        std::string toUID;
        
        Value srcVal; 
        Operation *toOp = metaTo.first;
        if (metaTo.second[0] == 'A') {
            int blkArgIdx = std::stoi(llvm::StringRef(metaTo.second).drop_front().str());
            srcVal = toOp->getRegion(0).front().getArgument(blkArgIdx);
        }
        else {
            int resultIdx = std::stoi(metaTo.second);
            srcVal = toOp->getResult(resultIdx);
        }
        toUID = toOp->getUID() + "_" + metaTo.second;
        if (srcVal.getType().isa<RankedTensorType>())
            srcVal = mutate::TensorReshape(srcVal, use->get(), toOp->getUID());
        use->set(srcVal);

        if (failed(verify(useOp->getParentOfType<FuncOp>()))) {
            llvm::errs() << "mlir-mutate(failed): replacing " << FromUID << " with " << toUID << "\n";
            return false;
        }

        llvm::errs() << "opreplaced " << FromUID << "," << toUID << "\n";
        useList.pop_back();
    }
    return true;
}

Operation* mutate::getLocalConstantOp(Operation *op, Value refV) {
    std::string typeStr;
    Type t = refV.getType();
    if (!t.isIntOrFloat()) // TODO: create the constant in complex types, like vector
        return nullptr;

    Operation *parentOp = op->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
    for (Operation *_op: traverseNestedOp(parentOp)) {
        if (_op->getUID().find("C1") != std::string::npos) {
            if (_op->getResult(0).getType() == refV.getType())
                return _op;
        }
    }

    OpBuilder builder(parentOp->getRegion(0));
    Operation *cop;
    
    unsigned width = t.getIntOrFloatBitWidth();
    if (t.isInteger(width)) {
        typeStr = "C1i" + std::to_string(width);
        cop = builder.create<mlir::ConstantIntOp>(builder.getUnknownLoc(), 1, t);
    }
    else {
        typeStr = "C1f" + std::to_string(width);
        cop = builder.create<mlir::ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat(1.0), t.cast<FloatType>());
    }
    cop->setUID(typeStr);

    return cop;
}

std::vector<Operation*> mutate::traverseNestedOp(Operation* startop, 
                                                 bool insertStartOp/*=false*/,
                                                 bool excludeIsolatedFromAbove/*=false*/ ) {
    std::vector<Operation*> ops;

    if (insertStartOp)
        ops.push_back(startop);
  
    if (startop->getNumRegions() == 0)
        return ops;

    for (int i=0; i<startop->getNumRegions(); ++i){
        for (auto &op: startop->getRegion(i).getOps()) {
            ops.push_back(&op);
            if (excludeIsolatedFromAbove && op.isKnownIsolatedFromAbove())
                continue;

            std::vector<Operation*> nestedops = 
                traverseNestedOp(&op, false, excludeIsolatedFromAbove);
            if (!nestedops.empty())
                ops.insert(ops.end(), nestedops.begin(), nestedops.end());
        }
    }
    return ops;
}

void mutate::updateUID(Operation* srcOp, Operation* dstOp, std::string mode) {
    
    std::vector<Operation*> srcNest, dstNest;
    srcNest = traverseNestedOp(srcOp, true, false);
    dstNest = traverseNestedOp(dstOp, true, false);
    assert(srcNest.size() == dstNest.size());

    std::vector<unsigned> cntVec(srcNest.size());
    ModuleOp m = srcOp->getParentOfType<ModuleOp>();

    for (unsigned i=0; i<srcNest.size(); i++) {
        std::string targetUID = srcNest[i]->getUID() + "." + mode;

        unsigned cnt = 0;
        for (Operation *_op: traverseNestedOp(m)) {
            if (_op->getUID().find(targetUID) != std::string::npos)
                cnt++;
        }
        dstNest[i]->setUID(targetUID + std::to_string(cnt+1));
    }
}

bool mutate::replaceUnfulfillOperands(Operation *op) {
    FuncOp f = op->getParentOfType<FuncOp>();
    for (auto &oprd : op->getOpOperands()) {
        std::string fromUID = op->getUID() + "_OP" + std::to_string(oprd.getOperandNumber());
        std::string toUID;
        std::pair<Operation*, std::string> metaV =
            randValueBeforeOp(f, op, oprd.get());
        
        if (metaV.first == nullptr) {
            llvm::errs() << "Operation failed. Cannot find an proper value for " << fromUID <<
            "<" << oprd.get().getType() << ">\n";
            return false;
        }

        Operation *toOp = metaV.first;
        Value srcVal;
        if (metaV.second[0] == 'A') {
            int blkArgIdx = std::stoi(llvm::StringRef(metaV.second).drop_front().str());
            srcVal = toOp->getRegion(0).front().getArgument(blkArgIdx);
        }
        else if (metaV.second[0] == 'C')
            srcVal = toOp->getResult(0);
        else {
            int resultIdx = std::stoi(metaV.second);
            srcVal = toOp->getResult(resultIdx);
        }
        toUID = toOp->getUID() + "_" + metaV.second;
        if (srcVal.getType().isa<RankedTensorType>())
            srcVal = mutate::TensorReshape(srcVal, oprd.get(), toOp->getUID());
        oprd.set(srcVal);
        
        llvm::errs()<<"opreplaced "<< fromUID << "," << toUID << "\n";
    }
    return true;
}

void mutate::useResult(Operation *op) {
    DominanceInfo dom = DominanceInfo(op);
    FuncOp f = op->getParentOfType<FuncOp>();
    std::vector<OpOperand*> oprdvec[2];

    /**
     * We don't want operations that are isolated from current op
     **/
    for (Operation *_op : traverseNestedOp(f, false, true)) {
        if (_op->getUID()[0] == 'R') // Do not use operations created for reshaping the tensor during repairing
            continue;
        if (_op->getUID().find("nop") != std::string::npos)
            continue;
        if (dom.properlyDominates(op, _op) == false)
            continue;

        for (auto &_oprd : _op->getOpOperands()) {
            for (auto opre : op->getResults()) {
                unsigned int resultIdx = opre.getResultNumber();
                if (_oprd.get() == opre)
                    continue;
                
                if (_oprd.get().getType() == opre.getType())
                    oprdvec[resultIdx].push_back(&_oprd);
            }
        }
    }

    for (auto opre : op->getResults()) {
        unsigned int resultIdx = opre.getResultNumber();
        std::string fromUID = op->getUID() + "_" + std::to_string(resultIdx);
        if (oprdvec[resultIdx].empty()) {
            llvm::errs() << "Cannot use result " << fromUID << "\n";
            continue;
        }

        std::uniform_int_distribution<> randIdx(0, oprdvec[resultIdx].size()-1);
        OpOperand *dstOprd = oprdvec[resultIdx][randIdx(gen)];
        std::string toUID = dstOprd->getOwner()->getUID() + "_OP" + std::to_string(dstOprd->getOperandNumber());
        dstOprd->set(opre);
        llvm::errs() << "opreplaced " << toUID << "," << fromUID << "\n";
    }
}

Operation* mutate::walkExact(std::string opDesc, std::string &UID, ModuleOp &m) {
    unsigned count = 0;
    for (auto *op : traverseNestedOp(m)) {
        if (op->getUID().empty())
            continue;
        count += 1;
        if (opDesc[0] == 'U' || opDesc[0] == 'R') {
            if (opDesc == op->getUID()) {
                UID = opDesc;
                return op;
            }
        }
        else {
            if (count == std::stoul(opDesc)) {
                UID = op->getUID();
                if (op->getUID()[0] != 'U') {
                    count--;
                    continue;
                }
                if (op->isKnownTerminator()) {
                    count--;
                    continue;
                }
                return op;
            }
        }
    }
    return nullptr;
}

Operation* mutate::walkCollect(std::string opDesc, std::string &UID, ModuleOp &m) {
    unsigned count = 0;
    for (auto *op : traverseNestedOp(m)) {
        if (op->getUID().empty())
            continue;
        count += 1;
        if (opDesc[0] == 'U') {
            llvm::StringRef ID = op->getUID();
            if (ID.find(".d") != StringRef::npos) continue; // Cannot be a deleted operation

            llvm::StringRef IDBase = ID.split('.').first;
            llvm::StringRef targetBase = llvm::StringRef(opDesc).split('.').first;
            if (IDBase.equals(targetBase)) {
                UID = opDesc;
                return op;
            }
        }
        else {
            if (count == std::stoul(opDesc)) {
                if (op->getUID()[0] != 'U') {
                    count--;
                    continue;
                }
                if (op->isKnownTerminator()) {
                    count--;
                    continue;
                }
                UID = op->getUID();
                return op;
            }
        }
    }
    return nullptr;
}

Operation* mutate::walkPosition(std::string opDesc, std::string &UID, ModuleOp &m) {
    unsigned count = 0;
    for (auto *op : traverseNestedOp(m)) {
        if (op->getUID().empty())
            continue;
        count += 1;
        if (opDesc[0] == 'U') {
            std::string ID = op->getUID();
            if ((ID.compare(opDesc) == 0) ||
                (ID.compare(opDesc + ".d") == 0)  ) {
                UID = opDesc;
                return op;
            }
        }
        else {
            if (count == std::stoul(opDesc)) {
                if (op->getUID()[0] != 'U') {
                    count--;
                    continue;
                }
                if (op->isKnownTerminator()) {
                    count--;
                    continue;
                }
                UID = op->getUID();
                // if (UID.empty())
                //     return nullptr;
                return op;
            }
        }
    }
    return nullptr;
}

Operation* mutate::insertNOP(Operation *refOp) {
    assert(refOp->getParentRegion());

    OpBuilder builder(refOp);
    auto nop = builder.create<mlir::ConstantIntOp>(builder.getUnknownLoc(), 0, 32);
    nop.getOperation()->setUID(refOp->getUID() + ".d");
    return nop.getOperation();
}

void mutate::OperandReplace::runOnOperation() {
    std::string dummy;
    ModuleOp module = getOperation();
    // decompose destination description into inst and operand
    StringRef dstOpDesc = (StringRef(dstDesc)).rsplit('_').first;
    StringRef dstOperandDesc = (StringRef(dstDesc)).rsplit('_').second;
    assert(dstOperandDesc.find("OP") != StringRef::npos && "Not a valid operand description!");
    unsigned operandIdx = std::stoul(dstOperandDesc.drop_front(2).str());// remove "OP"
    Operation *dstOp = walkExact(dstOpDesc.str(), dummy, module);
    // if (dstOp == NULL)
    //     return -1;
    // if (operandIdx >= dstOp->getNumOperands())
    //     return -2;
    OpOperand &dstOperand = dstOp->getOpOperand(operandIdx);

    StringRef srcOpDesc = (StringRef(srcDesc)).rsplit('_').first;
    StringRef srcResultDesc = (StringRef(srcDesc)).rsplit('_').second;
    Operation *srcOp = walkExact(srcOpDesc.str(), dummy, module);

    Value srcVal;
    if (srcResultDesc[0] == 'A') {
        int blkArgIdx = std::stoul(srcResultDesc.drop_front().str());
        srcVal = srcOp->getRegion(0).front().getArgument(blkArgIdx);
    }
    else {
        unsigned srcResultIdx = std::stoul(srcResultDesc.str());
        srcVal = srcOp->getResult(srcResultIdx);
    }

    if (dstOperand.get().getType().isa<RankedTensorType>())
        srcVal = mutate::TensorReshape(srcVal, dstOperand.get(), srcOpDesc.str());

    dstOperand.set(srcVal);
    // TODO: constant
    
    llvm::errs()<<"opreplaced "<< dstDesc << "," << srcDesc << "\n";
}

void mutate::Cut::runOnOperation() {
    // Get the current operation being operated on.
    ModuleOp module = getOperation();
    std::string rUID;
    Operation* op = walkExact(op1, rUID, module);
    if (op == nullptr) {
        llvm::errs() << "cut failed. Cannot find " << op1 << "\n";
        signalPassFailure();
        return;
    }

    FuncOp f = op->getParentOfType<FuncOp>();
    if (!op->use_empty()) {
        for (OpResult v : op->getResults()) {
            std::pair<Operation*, std::string> metaV =
                randValueBeforeOp(f, op, v);
            
            if(metaV.first == nullptr) {
                llvm::errs() << "cut failed. Cannot find result replacement for " <<
                    rUID+"_"+std::to_string(v.getResultNumber()) << "\n";
                signalPassFailure();
                return;
            }
                
            if (!replaceAllUsesWithReport(v, metaV)) {
                signalPassFailure();
                return;
            }
        }
    }
    insertNOP(op);
    op->erase();
    llvm::errs() << "cut " << rUID << "\n";
}

void mutate::Insert::runOnOperation() {
    // Get the current operation being operated on.
    ModuleOp module = getOperation();
    std::string srcUID, dstUID;
    Operation *srcOp = walkCollect(srcDesc, srcUID, module);
    Operation *dstOp = walkPosition(dstDesc, dstUID, module);
    if (srcOp == nullptr or dstOp == nullptr) {
        llvm::errs()<<"insertion failed. Cannot find/use ";
        if (srcOp == nullptr) llvm::errs() << srcDesc << " ";
        if (dstOp == nullptr) llvm::errs() << dstDesc << " ";
        llvm::errs() << "\n";
        signalPassFailure();
        return;
    }

    OpBuilder builder(dstOp);
    Operation *inserted = builder.clone(*srcOp);
    inserted->setUID(srcOp->getUID());
    updateUID(srcOp, inserted, "i");
    if (!replaceUnfulfillOperands(inserted)) {
        signalPassFailure();
        return;
    }

    // Rename the symbol name if it exist. Primarily for function operations
    // TODO: implement a better renaming scheme
    auto nameAttr = inserted->getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName());
    if (nameAttr != nullptr) {
        std::string newAttrStr = nameAttr.getValue().str() + "_" + inserted->getUID();
        inserted->setAttr(mlir::SymbolTable::getSymbolAttrName(), 
                          StringAttr::get(StringRef(newAttrStr), inserted->getContext()) );
    }

    if (!(srcOp->use_empty()) && !noResult)
        mutate::useResult(inserted);

    llvm::errs()<<"inserted " << dstUID << "," << srcUID << "\n";
}

void mutate::Name::runOnOperation() {
    ModuleOp m = getOperation();
    int cnt = 0;
    std::string uid;
    for (auto *op: traverseNestedOp(m)) {
        uid = "U" + std::to_string(cnt);
        cnt++;
        op->setUID(uid);
    }
}

void mutate::List::runOnOperation() {
    ModuleOp m = getOperation();
    int cnt = 0;
    for (auto *op: traverseNestedOp(m)) {
        if (op->getUID().empty())
            continue;
        if (op->getUID()[0] != 'U') // skip tensor reshape operation
            continue;
        if (op->getUID().find("nop") != std::string::npos) // skip the remain of a deleted operation
            continue;
        if (op->isKnownTerminator())
            continue;
        cnt++;
    }
    llvm::errs() << cnt << "\n";
}
