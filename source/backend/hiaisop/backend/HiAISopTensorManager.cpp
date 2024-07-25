#include "HiAISopTensorManager.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
static HiAI_SingleOp_Format ConvertFormat(MNN_DATA_FORMAT dimensionFormat) {
    switch (dimensionFormat) {
        case MNN_DATA_FORMAT_NCHW:
            return HIAI_SINGLEOP_FORMAT_NCHW;
        case MNN_DATA_FORMAT_NHWC:
            return HIAI_SINGLEOP_FORMAT_NHWC;
        case MNN_DATA_FORMAT_NC4HW4:
            return HIAI_SINGLEOP_FORMAT_NC4HW4;
        default:
            return HIAI_SINGLEOP_FORMAT_RESERVED;
    }
}

static HiAI_SingleOp_DataType ConvertDataType(halide_type_t type) {
    if (type == halide_type_of<float>()) {
        return HIAI_SINGLEOP_DT_FLOAT;
    } else if (type == halide_type_t(halide_type_float, 16)) {
        return HIAI_SINGLEOP_DT_FLOAT16;
    } else {
        return HIAI_SINGLEOP_DT_UNDEFINED;
    }
}

static HiAI_SingleOpTensorDesc* CreateSopTensorDesc(Tensor* t, bool isVirtual) {
    size_t dimNum = t->dimensions();
    int64_t dims[dimNum];
    for (int i = 0; i < dimNum; i++) {
        dims[i] = t->length(i);
    }
    HiAI_SingleOp_Format format = HIAI_SINGLEOP_FORMAT_RESERVED;
    HiAI_SingleOp_DataType dataType = HIAI_SINGLEOP_DT_UNDEFINED;
    if (!isVirtual) {
        format = ConvertFormat(TensorUtils::getDescribe(t)->dimensionFormat);
        if (format == HIAI_SINGLEOP_FORMAT_RESERVED) {
            return nullptr;
        }
        dataType = ConvertDataType(t->getType());
        if (dataType == HIAI_SINGLEOP_DT_UNDEFINED) {
            return nullptr;
        }
    }
    return SingleOpTensorDesc_Create(dims, dimNum, dataType,
        format, isVirtual);
}

HiAISopMemObj::HiAISopMemObj(Tensor* nativeTensor, bool isVirtual) {
    mTensorDesc = CreateSopTensorDesc(nativeTensor, isVirtual);
    mTensor = SingleOpTensor_CreateFromTensorDesc(mTensorDesc);
    auto buf = SingleOpTensor_GetBuffer(mTensor);
    auto desc = SingleOpTensor_GetTensorDesc(mTensor);
    mPoint = MemChunk(SingleOpBuffer_GetData(buf), 0);
    mSize = SingleOpTensorDesc_GetByteSize(desc);
}
    
HiAISopMemObj::~HiAISopMemObj() {
    SingleOpTensorDesc_Destroy(&mTensorDesc);
    SingleOpTensor_Destroy(&mTensor);
}

bool HiAISopMemObj::reallocate() {
    if (mTensor != nullptr) {
        SingleOpTensor_Destroy(&mTensor);
    }
    mTensor = SingleOpTensor_CreateFromTensorDesc(mTensorDesc);
    if (mTensor == nullptr) {
        MNN_ERROR("Create sop tensor failed\n");
        return false;
    }
    auto buf = SingleOpTensor_GetBuffer(mTensor);
    auto desc = SingleOpTensor_GetTensorDesc(mTensor);
    mPoint = MemChunk(SingleOpBuffer_GetData(buf), 0);
    mSize = SingleOpTensorDesc_GetByteSize(desc);
    return true;
}

Backend::MemObj* HiAISopTensorManager::onAlloc(Tensor* t, bool isVirtual) {
    Backend::MemObj* memObj = nullptr;
    if (mMemObjContainer.find(t) != mMemObjContainer.end()) {
        memObj = mMemObjContainer[t];
    } else {
        memObj = new HiAISopMemObj(t, isVirtual);
        mMemObjContainer[t] = memObj;
    }
    MemChunk chunk = memObj->chunk();
    if (chunk.ptr()) {
        auto& buffer = t->buffer();
        buffer.host = chunk.ptr();
    }
    return memObj;
}

void HiAISopTensorManager::onRelease(Tensor* t) {
    if (mMemObjContainer.find(t) == mMemObjContainer.end()) {
        MNN_ERROR("MNN tensor has no hiai sop tensor obj\n");
        return;
    }
    auto memObj = mMemObjContainer[t];
    delete memObj;
    auto& buffer = t->buffer();
    buffer.host = nullptr;
    mMemObjContainer.erase(t);
    return;
}

HiAI_SingleOpTensorDesc* HiAISopTensorManager::getHiAISopTensorDesc(const Tensor* t) {
    if (mMemObjContainer.find(t) == mMemObjContainer.end()) {
        MNN_ERROR("MNN tensor has no hiai sop tensor obj\n");
        return nullptr;
    }
    auto memObj = static_cast<HiAISopMemObj*>(mMemObjContainer[t]);
    return memObj->sopTensorDesc();
}

HiAI_SingleOpTensor* HiAISopTensorManager::getHiAISopTensor(const Tensor* t) {
    if (mMemObjContainer.find(t) == mMemObjContainer.end()) {
        MNN_ERROR("MNN tensor has no hiai sop tensor obj\n");
        return nullptr;
    }
    auto memObj = static_cast<HiAISopMemObj*>(mMemObjContainer[t]);
    return memObj->sopTensor();
}

bool HiAISopTensorManager::onUpdateTensorMem(Tensor* t) {
    if (mMemObjContainer.find(t) == mMemObjContainer.end()) {
        MNN_ERROR("MNN tensor has no hiai sop tensor obj\n");
        return false;
    }
    auto memObj = static_cast<HiAISopMemObj*>(mMemObjContainer[t]);
    if (!memObj->reallocate()) {
        return false;
    }
    MemChunk chunk = memObj->chunk();
    if (chunk.ptr()) {
        auto& buffer = t->buffer();
        buffer.host = chunk.ptr();
    }
    return true;
}
}