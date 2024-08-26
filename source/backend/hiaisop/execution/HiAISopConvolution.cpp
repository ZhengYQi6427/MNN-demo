#include "HiAISopConvolution.hpp"
#include "HiAISopTensorManager.hpp"
#include "core/ConvolutionCommon.hpp"

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
    void* weightData = reinterpret_cast<void*>(const_cast<float*>(originWeight));
    if (weightData == nullptr) {
        MNN_ERROR("NULL weightData\n");
    }
    mWeight = SingleOpTensor_CreateFromConst(weightTensorDesc, weightData, originWeightSize * sizeof(float));
    if (mWeight == nullptr) {
        MNN_ERROR("create weight tensor failed\n");
    }
    SingleOpTensorDesc_Destroy(&weightTensorDesc);

    if (originBias != nullptr) {
        int64_t biasDims[] = {oc};
        auto biasTensorDesc = SingleOpTensorDesc_Create(biasDims, 1,
            HIAI_SINGLEOP_DT_FLOAT, HIAI_SINGLEOP_FORMAT_NCHW, false);
        void* biasData = reinterpret_cast<void*>(const_cast<float*>(originBias));
        mBias = SingleOpTensor_CreateFromConst(biasTensorDesc,
            biasData, originBiasSize * sizeof(float));
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
    HiAISopBackend* backend = static_cast<HiAISopBackend*>(this->backend());
    auto inputDesc = backend->getHiAISopTensorDesc(inputs[0]);
    auto outputDesc = backend->getHiAISopTensorDesc(outputs[0]);

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
        mMode, strides, dilations, pads, groups, padMode);

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
        setExecutor(SingleOpExecutor_CreateFusedConvolutionActivation(options, convOpDesc, actOpDesc,
            inputDesc, outputDesc, mWeight, mBias));
    } else {
        setExecutor(SingleOpExecutor_CreateConvolution(options, convOpDesc,
            inputDesc, outputDesc, mWeight, mBias));
    }

    // release tmp resource
    SingleOpOptions_Destroy(&options);
    SingleOpDescriptor_Destroy(&convOpDesc);
    SingleOpDescriptor_Destroy(&actOpDesc);

    if (executor() == nullptr) {
        MNN_ERROR("create sop executor failed\n");
        return NOT_SUPPORT;
    }
    // update output
    HiAI_SingleOp_Format outFormat = SingleOpTensorDesc_GetFormat(outputDesc);
    if (outFormat == HIAI_SINGLEOP_FORMAT_RESERVED) {
        if (SingleOpExecutor_UpdateOutputTensorDesc(executor(), 0, outputDesc) != 0) {
            MNN_ERROR("update output tensor desc failed\n");
            return NOT_SUPPORT;
        }
        if (!backend->onUpdateTensorMem(outputs[0])) {
            MNN_ERROR("update output tensor mem failed\n");
            return NOT_SUPPORT;
        }
    }
    backend->updateWorkspaceSize(SingleOpExecutor_GetWorkspaceSize(mExec));
    return NO_ERROR;
}

ErrorCode HiAISopConvolution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        MNN_ERROR("Inputs or output count of %s is invalid\n", name().c_str());
        return NOT_SUPPORT;
    }
    HiAISopBackend* backend = static_cast<HiAISopBackend*>(this->backend());
    if (backend == nullptr) {
        MNN_ERROR("NULL backend\n");
        return NOT_SUPPORT;
    }
    HiAI_SingleOpTensor* sopInputs[] = {backend->getHiAISopTensor(inputs[0])};
    HiAI_SingleOpTensor* sopOutputs[] = {backend->getHiAISopTensor(outputs[0])}
    if (SingleOpExecutor_Execute(executor(), sopInputs, 1, sopOutputs, 1) != 0) {
        MNN_ERROR("Execute [%s] failed\n", name().c_str());
        return NOT_SUPPORT;
    }
    return NO_ERROR;
}

HiAI_SingleOp_SupportStatus HiAISopConvolution::PreCheck(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    HiAI_SingleOp_SupportStatus ret = HIAI_SINGLEOP_UNSUPPORTED;
    auto inputDesc = CreateSopTensorDesc(inputs[0], false);
    auto outputDesc = CreateSopTensorDesc(outputs[0], true);

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
        mMode, strides, dilations, pads, groups, padMode);

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
        ret = SingleOpExecutor_PreCheckFusedConvolutionActivation(options, convOpDesc, actOpDesc,
            inputDesc, outputDesc, mWeight, mBias);
    } else {
        ret = SingleOpExecutor_PreCheckConvolution(options, convOpDesc,
            inputDesc, outputDesc, mWeight, mBias);
    }

    // release tmp resource
    SingleOpOptions_Destroy(&options);
    SingleOpDescriptor_Destroy(&convOpDesc);
    SingleOpDescriptor_Destroy(&actOpDesc);
    SingleOpTensorDesc_Destroy(&inputDesc);
    SingleOpTensorDesc_Destroy(&outputDesc);
    return ret;
}

namespace {
uint32_t GetSingleOpID()
{
    static std::atomic<int32_t> id(1);
    return (id++);
}
}

class HiAISopConvolutionCreator : public HiAISopBackend::HiAISopCreator {
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        auto conv2d = op->main_as_Convolution2D();
        std::string opName = "singleop_" + std::to_string(GetSingleOpID());
        if (op->name() != nullptr) {
            opName = op->name()->str();
        }
        // get weight and bias data
        const float* originWeight = nullptr;
        const float* originBias = nullptr;
        int originWeightSize = 0;
        int originBiasSize = 0;
        if (inputs.size() > 1) {
            originWeight = inputs[1]->host<float>();
            originWeightSize = inputs[1]->elementSize();
            if (inputs.size() > 2) {
                originBias = inputs[2]->host<float>();
                originBiasSize = inputs[2]->elementSize();
            }
        } else {
            std::shared_ptr<ConvolutionCommon::Int8Common> quanComm;
            if (conv2d->quanParameter() != nullptr) {
                quanComm = ConvolutionCommon::load(conv2d, backend, true, false);
                if (quanComm == nullptr) {
                    MNN_ERROR("Memory not enough, can't extract IDST Convolution: %s \n", opName.c_str());
                    return nullptr;
                }
                if (conv2d->quanParameter()->has_scaleInt) {
                    MNN_PRINT("[%s]: quantized conv does not supported\n", opName.c_str());
                    return nullptr;
                }
                // Back to float
                originWeight = quanComm->weightFloat.get();
                originWeightSize = quanComm->weightFloat.size();
            } else if (conv2d->weight() == nullptr || conv2d->bias() == nullptr) {
                MNN_ERROR("%s has no weight or bias. The model may be benchmark model, please revert the weight/bias firstly\n", opName.c_str());
                return nullptr;
            }

            if (originWeight == nullptr && conv2d->weight() != nullptr) {
                originWeight = conv2d->weight()->data();
                originWeightSize = conv2d->weight()->size();
            }
            if (originBias == nullptr && conv2d->bias() != nullptr) {
                originBias = conv2d->bias()->data();
                originBiasSize = conv2d->bias()->size();
            }
        }

        auto common = conv2d->common();
        auto opType = op->type();
        HiAISopConvolution* convExec = new HiAISopConvolution(common, backend, originWeight, originWeightSize,
            originBias, originBiasSize, opType, opName);
        auto preCheckRet = convExec->PreCheck(inputs, outputs);
        if (preCheckRet != HIAI_SINGLEOP_OPTIMIZED) {
            MNN_PRINT("HiAISopConvolutionCreator [%s] precheck failed: %d.\n", opName.c_str(), preCheckRet);
            return nullptr;
        }
        MNN_PRINT("HiAISopConvolutionCreator create: %s\n", opName.c_str());
        return convExec;
    }
};

REGISTER_HIAISOP_OP_CREATOR(OpType_Convolution, HiAISopConvolutionCreator);
REGISTER_HIAISOP_OP_CREATOR(OpType_ConvolutionDepthwise, HiAISopConvolutionCreator);
REGISTER_HIAISOP_OP_CREATOR(OpType_Deconvolution, HiAISopConvolutionCreator);
REGISTER_HIAISOP_OP_CREATOR(OpType_DeconvolutionDepthwise, HiAISopConvolutionCreator);

} // namespace MNN