#include "HiAISopConvolution.hpp"

namespace MNN {
static HiAI_SingleOp_ConvMode getHiAISopConvMode(OpType opType) {
    static std::map<OpType, HiAI_SingleOp_ConvMode> typeToConvModeMap = {
        {OpType_Convolution, HIAI_SINGLEOP_CONV_MODE_COMMON},
        {OpType_ConvolutionDepthwise, HIAI_SINGLEOP_CONV_MODE_DEPTHWISE},
        {OpType_Deconvolution, HIAI_SINGLEOP_CONV_MODE_TRANSPOSED},
        {OpType_DeconvolutionDepthwise, HIAI_SINGLEOP_CONV_MODE_TRANSPOSED},
    };
    if (typeToConvModeMap.find(opType) == typeToConvModeMap.end()) {
        return HIAI_SINGLEOP_CONV_MODE_COMMON;
    }
    return typeToConvModeMap[opType];
}

HiAISopConvolution::HiAISopConvolution(const Convolution2DCommon* common, Backend* backend,
    const float* originWeight, size_t originWeightSize, const float* originBias,
    size_t originBiasSize, OpType opType, std::string opName) : MNN::HiAISopExecution(backend, opName) {
    mMode = getHiAISopConvMode(opType);
    mCommon = common;
    // create weight and bias hiai sop tensor
    auto kw = common->kernelX();
    auto kh = common->kernelY();
    auto ic = common->inputCount();
    auto oc = common->outputCount();
    if (ic == 0) {
        ic = originWeightSize / oc / kw / kh;
    }
    int64_t weightDims[] = {oc, ic, kh, kw};
    if (mMode == HIAI_SINGLEOP_CONV_MODE_TRANSPOSED) {
        weightDims[0] = ic;
        weightDims[1] = oc;
    }
    auto weightTensorDesc = SingleOpTensorDesc_Create(weightDims, 4,
        HIAI_SINGLEOP_DT_FLOAT, HIAI_SINGLEOP_FORMAT_NCHW, false);
    mWeight = SingleOpTensor_CreateFromConst(weightTensorDesc,
        originWeight, originWeightSize);
    if (mWeight == nullptr) {
        MNN_ERROR("create weight tensor failed\n");
    }
    SingleOpTensorDesc_Destroy(&weightTensorDesc);

    if (originBias != nullptr) {
        int64_t biasDims[] = {oc};
        auto biasTensorDesc = SingleOpTensorDesc_Create(biasDims, 1,
            HIAI_SINGLEOP_DT_FLOAT, HIAI_SINGLEOP_FORMAT_NCHW, false);
        mBias = SingleOpTensor_CreateFromConst(biasTensorDesc,
            originBias, originBiasSize);
        if (mBias == nullptr) {
            MNN_ERROR("create bias tensor failed\n");
        }
        SingleOpTensorDesc_Destroy(&biasTensorDesc);
    }
    
}

HiAISopConvolution::~HiAISopConvolution() {
    SingleOpExecutor_Destroy(&mExec);
    // destroy weight and bias tensor
    SingleOpTensor_Destroy(&mWeight);
    SingleOpTensor_Destroy(&mBias);
}

ErrorCode HiAISopConvolution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto inputDesc = mBackend->GetHiAISopTensorDesc(inputs[0]);
    auto outputDesc = mBackend->GetHiAISopTensorDesc(outputs[0]);

    int64_t strides[] = {mCommon->strideY(), mCommon->strideX()};
    int64_t dilations[] = {mCommon->dilateY(), mCommon->dilateX()};
    int64_t pads[] = {mCommon->padY(), mCommon->padY(), mCommon->padX(), mCommon->padX()};
    int groups = mCommon->group();
    HiAI_SingleOp_PadMode padMode = HIAI_SINGLEOP_PAD_MODE_SPECIFIC;
    if (mCommon->padMode() == PadMode_VALID) {
        padMode = HIAI_SINGLEOP_PAD_MODE_VALID;
    } else if (mCommon->padMode() == PadMode_SAME) {
        padMode = HIAI_SINGLEOP_PAD_MODE_SAME;
    }
    HiAI_SingleOpDescriptor* convOpDesc = SingleOpDescriptor_CreateConvolution(
        mMode, stride, dilations, pads, groups, padMode);

    HiAI_SingleOpDescriptor* actOpDesc = nullptr;
    auto relu = mCommon->relu();
    auto relu6 = mCommon->relu6();
    if (relu || relu6) {
        HiAI_SingleOp_ActivationType actType = relu ?
            HIAI_SINGLEOP_ACTIVATION_TYPE_RELU : HIAI_SINGLEOP_ACTIVATION_TYPE_RELU6;
        actOpDesc = SingleOpDescriptor_CreateActivation(actType, 0);
    }

    auto options = SingleOpOptions_Create();
    if (actOpDesc != nullptr) {
        mExec = SingleOpExecutor_CreateFusedConvolutionActivation(options, convOpDesc, actOpDesc,
            inputDesc, outputDesc, mWeight, mBias);
    } else {
        mExec = SingleOpExecutor_CreateConvolution(options, convOpDesc,
            inputDesc, outputDesc, mWeight, mBias);
    }

    // release tmp resource
    SingleOpOptions_Destroy(&options);
    SingleOpDescriptor_Destroy(&convOpDesc);
    SingleOpDescriptor_Destroy(&actOpDesc);

    if (mExec == nullptr) {
        MNN_ERROR("create sop executor failed\n");
        return NOT_SUPPORT;
    }
    // update output
    HiAI_SingleOp_Format outFormat = SingleOpTensorDesc_GetFormat(outputDesc);
    if (outFormat == HIAI_SINGLEOP_FORMAT_RESERVED) {
        if (SingleOpExecutor_UpdateOutputTensorDesc(mExec, 0, outputDesc) != 0 ||
            !mBackend->onUpdateTensorMem(outputs[0])) {
            MNN_ERROR("update output failed\n");
            return NOT_SUPPORT;
        }
    }
    mBackend->updateWorkspaceSize(SingleOpExecutor_GetWorkspaceSize(mExec));
    return NO_ERROR;
}

ErrorCode HiAISopConvolution::Init(void* workspace) {
    if (SingleOpExecutor_Init(mExec, mBackend()->getWorkspace(), this->workspaceSize()) != 0) {
        return NOT_SUPPORT;
    }
    return NO_ERROR;
}

ErrorCode HiAISopConvolution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (inputs.size() != 1 || outputs.size()) {
        MNN_ERROR("Inputs or output count of %s is invalid\n", mOpName.c_str());
        return NOT_SUPPORT;
    }
    HiAI_SingleOpTensor* sopInputs[] = {mBackend->getHiAISopTensor(inputs[0])};
    HiAI_SingleOpTensor* sopOutputs[] = {mBackend->getHiAISopTensor(outputs[0])}
    if (SingleOpExecutor_Execute(mExec, sopInputs, 1, sopOutputs, 1) != 0) {
        MNN_ERROR("Execute [%s] failed\n", mOpName.c_str());
        return NOT_SUPPORT;
    }
    return NO_ERROR;
}

class HiAISopConvolutionCreator : public HiAISopBackend::HiAISopCreator {
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto conv2d = op->main_as_Convolution2D();
        if (inputs.size() != 1 || outputs.size() != 1) {
            MNN_PRINT("[%s]: multi input or output does not supported\n", op->name()->c_str());
            return nullptr;
        }
        if (conv2d->quanParameter != nullptr) {
            MNN_PRINT("[%s]: quantized conv does not supported\n", op->name()->c_str());
            return nullptr;
        }
        // get weight and bias data
        const float* originWeight = nullptr;
        const float* originBias = nullptr;
        int originWeightSize = 0;
        int originBiasSize = 0;
        if (conv2d->external() && conv2d->external()->size() > 1) {
            std::unique_ptr<Tensor> externalWeightTensor, externalBiasTensor;
            bool res = OpCommonUtils::loadConvData(backend, op, externalWeightTensor, externalBiasTensor,
                originWeightSize, originBiasSize);
            if (!res) {
                MNN_ERROR("[%s] load external weight or bias failed\n", op->name()->c_str());
                return nullptr;
            }
            originWeight = externalWeightTensor->host<float>();
            originBias = externalBiasTensor->host<float>();
        } else {
            if (conv2d->weight() == nullptr) {
                MNN_ERROR("[%s] has no weight\n", op->name()->c_str());
                return nullptr;
            }
            originWeight = conv2d->weight()->data();
            originWeightSize = conv2d->weight()->size();
            if (conv2d->bias() != nullptr) {
                originBias = conv2d->bias()->data();
                originBiasSize = conv2d->bias()->size();
            }
        }

        auto common = conv2d->common();
        auto opType = op->type();
        HiAISopConvolution* convExec = new HiAISopConvolution(common, backend, originWeight, originWeightSize,
            originBias, originBiasSize, opType, op->name()->str());
        if (convExec->Precheck() != HIAI_SINGLEOP_OPTIMIZED) {
            MNN_PRINT("Op [%s] is not optimized by HiAI.\n", op->name()->c_str());
            return nullptr;
        }
        return convExec;
    }
};

REGISTER_HIAISOP_OP_CREATOR(OpType_Convolution, HiAISopConvolutionCreator);
REGISTER_HIAISOP_OP_CREATOR(OpType_ConvolutionDepthwise, HiAISopConvolutionCreator);
REGISTER_HIAISOP_OP_CREATOR(OpType_Deconvolution, HiAISopConvolutionCreator);
REGISTER_HIAISOP_OP_CREATOR(OpType_DeconvolutionDepthwise, HiAISopConvolutionCreator);

} // namespace MNN