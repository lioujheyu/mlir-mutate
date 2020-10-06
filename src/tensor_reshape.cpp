#include "mutate.h"

Value& mutate::TensorReshape(Value& from, const Value& to, std::string fromDesc) {
    // Check if value `from` is absolutely larger or smaller than value `to`
    auto fromShape = from.getType().cast<TensorType>().getShape();
    auto fromRank = from.getType().cast<TensorType>().getRank();
    auto toShape = to.getType().cast<TensorType>().getShape();
    auto toRank = to.getType().cast<TensorType>().getRank();
    auto toElementType = to.getType().cast<TensorType>().getElementType();
    int fromShapeShift = 0;

    std::vector<int64_t> extFromShape, extToShape;
    if (fromRank > toRank) {
        extFromShape = fromShape.vec();
        extToShape = std::vector<int64_t>(fromRank - toRank, 1);
        extToShape.insert(extToShape.end(), toShape.begin(), toShape.end());
    }
    else if (fromRank < toRank) {
        fromShapeShift = toRank - fromRank;
        extFromShape = std::vector<int64_t>(toRank - fromRank, 1);
        extFromShape.insert(extFromShape.end(), fromShape.begin(), fromShape.end());
        extToShape = toShape.vec();
    }
    else { // fromRank == toRank
        extFromShape = fromShape.vec();
        extToShape = toShape.vec();
    }

    enum class ReshapeOp {
        mhloBroadcast = 0,
        mhloReshape = 1,
        mhloSlice = 2,
        mhloPad = 3
    };
    
    std::vector<std::pair<ReshapeOp, std::vector<int64_t>>> reshapeOps; /*op, dim index of from*/
    
    for (int dim=extFromShape.size()-1; dim>=0; --dim) {
        ReshapeOp rop;
        if (dim < fromShapeShift) {
            rop = ReshapeOp::mhloBroadcast;
        }
        else { // dim < both fromRank and toRank
            if (extFromShape[dim] > extToShape[dim]) {
                rop = ReshapeOp::mhloSlice;
            }
            else if (extFromShape[dim] < extToShape[dim]) { // fromShape[dim] < toShape[dim]
                rop = ReshapeOp::mhloPad;
            }
            else // fromShape[dim] == toShape[dim]: do nothing
                continue;
        }

        if (!reshapeOps.empty()) {
            if (reshapeOps.back().first == rop) {
                reshapeOps.back().second.push_back(dim - fromShapeShift);
                continue;
            }
        }
        reshapeOps.push_back(std::make_pair(
            rop, std::vector<int64_t>(1, dim - fromShapeShift)));
    }
    //mhlo.reshape will be computed separately
    if (fromRank > toRank)
        reshapeOps.push_back(std::make_pair(
            ReshapeOp::mhloReshape, std::vector<int64_t>(0)));

    if (reshapeOps.empty()) // no need to reshape
        return from;
    
    OpBuilder builder(from.getContext());
    if (from.getDefiningOp())
        builder.setInsertionPointAfter(from.getDefiningOp());
    else { // Block argument
        builder.setInsertionPointToStart(from.getParentBlock());
    }

    Value& currentValue = from;
    std::vector<int64_t> currentShape = fromShape.vec();
    RankedTensorType type;
    for (std::pair<ReshapeOp, std::vector<int64_t>> opEntry : reshapeOps) {
        switch (opEntry.first) {
        case ReshapeOp::mhloBroadcast: {
            std::vector<int64_t> broadcast_sizes(0);
            for (int64_t i : opEntry.second) {
                broadcast_sizes.push_back(toShape[i + fromShapeShift]);
            }
            std::reverse(broadcast_sizes.begin(), broadcast_sizes.end());
            type = RankedTensorType::get(broadcast_sizes.size(), builder.getIntegerType(64));
            Operation *broadcastop = builder.create<mlir::mhlo::BroadcastOp>(builder.getUnknownLoc(), 
                RankedTensorType::get(toShape, toElementType),   // resultType
                currentValue,                                    // operand 0
                DenseIntElementsAttr::get(type, broadcast_sizes) // broadcast_sizes
            );
            currentValue = broadcastop->getResult(0);
            broadcastop->setUID("R" + fromDesc);
            break; }
        case ReshapeOp::mhloReshape: {
            Operation *reshapeop = builder.create<mhlo::ReshapeOp> (builder.getUnknownLoc(),
                RankedTensorType::get(toShape, toElementType), // resultType
                currentValue
            );
            // This will be the last operation
            currentValue = reshapeop->getResult(0);
            reshapeop->setUID("R" + fromDesc);
            break; }
        case ReshapeOp::mhloSlice: {
            std::vector<int64_t> start_indices(0), limit_indices(0), strides(0);
            for (int fromDim=0; fromDim<fromRank; ++fromDim) {
                strides.push_back(1);
                if (std::find(opEntry.second.begin(), opEntry.second.end(), fromDim) == opEntry.second.end()) {
                    start_indices.push_back(0);
                    limit_indices.push_back(currentShape[fromDim]);
                    continue;
                }

                //example: 5->2: 1.5 ~ 3.5 = 1 ~ 3: (\) x x (\) (\) 
                int dim = fromDim + fromShapeShift;
                start_indices.push_back((int)std::floor(
                    (float)extFromShape[dim]/2 - (float)extToShape[dim]/2));
                limit_indices.push_back((int)std::floor(
                    (float)extFromShape[dim]/2 + (float)extToShape[dim]/2));
                currentShape[fromDim] = extToShape[dim];
            }

            type = RankedTensorType::get(currentShape.size(), builder.getIntegerType(64));
            Operation *sliceop = builder.create<mhlo::SliceOp>(builder.getUnknownLoc(), 
                RankedTensorType::get(ArrayRef<int64_t>(currentShape), toElementType), // resultType
                currentValue,                                                          // operand 0, input
                DenseIntElementsAttr::get(type, start_indices),
                DenseIntElementsAttr::get(type, limit_indices),
                DenseIntElementsAttr::get(type, strides)
            );
            builder.setInsertionPointAfter(sliceop);
            currentValue = sliceop->getResult(0);
            sliceop->setUID("R" + fromDesc);
            break; }
        case ReshapeOp::mhloPad: {
            Operation *constop, *padop;
            std::string typeStr;
            unsigned width = toElementType.getIntOrFloatBitWidth();
            if (toElementType.isInteger(width)) {
                typeStr = "C0i" + std::to_string(width);
                constop = builder.create<mhlo::ConstOp>(
                    builder.getUnknownLoc(), 
                    DenseElementsAttr::get(RankedTensorType::get({}, toElementType.cast<IntegerType>()),
                                           IntegerAttr::get(toElementType.cast<IntegerType>(), 0)));
            }
            else {
                typeStr = "C0f" + std::to_string(width);
                constop = builder.create<mhlo::ConstOp>(
                    builder.getUnknownLoc(), 
                    DenseElementsAttr::get(RankedTensorType::get({}, toElementType.cast<FloatType>()),
                                           FloatAttr::get(toElementType.cast<FloatType>(), 0.0)));
            }
            constop->setUID(typeStr);
            builder.setInsertionPointAfter(constop);

            std::vector<int64_t> edge_padding_low(0), edge_padding_high(0), interior_padding(0);
            for (int fromDim=0; fromDim<fromRank; ++fromDim) {
                interior_padding.push_back(0);
                if (std::find(opEntry.second.begin(), opEntry.second.end(), fromDim) == opEntry.second.end()) {
                    edge_padding_low.push_back(0);
                    edge_padding_high.push_back(0);
                    continue;
                }

                //example: 2->5: padleft(1.5):1 padright(1.5):2 -> 0 x x 0 0
                int dim = fromDim + fromShapeShift;
                edge_padding_low.push_back((int)std::floor(((float)extToShape[dim] - extFromShape[dim])/2));
                edge_padding_high.push_back((int)std::round(((float)extToShape[dim] - extFromShape[dim])/2));
                currentShape[fromDim] = extToShape[dim];
            }

            type = RankedTensorType::get(currentShape.size(), builder.getIntegerType(64));
            padop = builder.create<mlir::mhlo::PadOp>(builder.getUnknownLoc(), 
                RankedTensorType::get(ArrayRef<int64_t>(currentShape), toElementType), // resultType
                currentValue,                                                          // operand 0, input
                constop->getResult(0),                                                 // operand 1, padding value
                DenseIntElementsAttr::get(type, edge_padding_low),
                DenseIntElementsAttr::get(type, edge_padding_high),
                DenseIntElementsAttr::get(type, interior_padding)
            );
            builder.setInsertionPointAfter(padop);
            currentValue = padop->getResult(0);
            padop->setUID("R" + fromDesc);
            break; }
        }
    }
    return currentValue;
}