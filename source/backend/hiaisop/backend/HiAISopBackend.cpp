#include "HiAISopBackend.hpp"
#include "core/TensorUtils.hpp"
#include "HiAISopSymbol.hpp"
#include "HiAISopTensorManager.hpp"
#include "HiAISopExecution.hpp"

namespace MNN {
void registerHiAISopOps();

static inline std::map<OpType, HiAISopBackend::HiAISopCreator*>* getHiAISopCreatorContainer() {
    static std::once_flag fg;
    static std::map<OpType, HiAISopBackend::HiAISopCreator*>* ret = nullptr;
    std::call_once(fg, [&] { ret = new std::map<OpType, HiAISopBackend::HiAISopCreator*>; });
    return ret;
}

HiAISopBackend::~HiAISopBackend() {
    SingleOpBuffer_Destroy(&mWorkspaceBuffer);
}

HiAISopBackend::HiAISopBackend(const CPURuntime* runtime, BackendConfig::MemoryMode memory)
     : CPUBackend(runtime, BackendConfig::Precision_Low, memory, MNN_FORWARD_CPU_EXTENSION) {
    mExecContainer.clear();
    mTensorManager = std::make_shared<HiAISopTensorManager>();
}

Execution* HiAISopBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op) {
    auto creatorContainer = getHiAISopCreatorContainer();
    auto iter = creatorContainer->find(op->type());
    if (iter != creatorContainer->end()) {
        auto exec = iter->second->onCreate(inputs, outputs, op, this);
        if (exec != nullptr) {
            addSopExecution(exec, inputs, outputs);
            // MNN_PRINT("HiAISop create execution for type: [%s]\n", MNN::EnumNameOpType(op->type()));
            return exec;
        }
    }
    // MNN_PRINT("HiAISop don't support type: [%s]\n", MNN::EnumNameOpType(op->type()));
    return CPUBackend::onCreate(inputs, outputs, op);
}

// 为所有Tensor分配内存
Backend::MemObj* HiAISopBackend::onAcquire(const Tensor* nativeTensor, StorageType storageType) {
    // 普通算子 -- CPUBackend::onAcquire
    // 单算子 -- mTensorManager
    auto sopSt = this->getHiAISopStorageType(nativeTensor);
    if (sopSt == CPU_USE_ONLY) {
        return CPUBackend::onAcquire(nativeTensor, storageType);
    }
    auto originMem = TensorUtils::getDescribeOrigin(nativeTensor)->mem.get();
    size_t size = CPUBackend::getTensorSize(nativeTensor, true);
    auto dest = const_cast<Tensor*>(nativeTensor);
    if (originMem != nullptr) {
        if (static_cast<HiAISopMemObj*>(originMem)->size() >= ALIGNED_SIZE(size)) {
            return originMem;
        } else {
            MNN_PRINT("HiAISop MemObj size not enough: [%zu] < [%zu]\n",
                static_cast<HiAISopMemObj*>(originMem)->size(),
                ALIGNED_SIZE(size));
            mTensorManager->onRelease(dest);
            TensorUtils::getDescribeOrigin(dest)->mem = nullptr;
        }
    }
    MNN_ASSERT(storageType == DYNAMIC || storageType == DYNAMIC_SEPERATE);
    TensorUtils::getDescribe(dest)->memoryType = Tensor::InsideDescribe::MEMORY_OUTSIDE;
    TensorUtils::getDescribe(dest)->usage = TensorUsage::HIAI_SOP;
    return mTensorManager->onAlloc(dest, sopSt == HIAI_SOP_USE_ONLY);
}

bool HiAISopBackend::onClearBuffer() {
    mTensorManager->releaseAll();
    return CPUBackend::onClearBuffer();
}

void HiAISopBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) {
    return;
}

void HiAISopBackend::onResizeBegin() {
    mTensorManager.reset(new HiAISopTensorManager);
    CPUBackend::onResizeBegin();
}

ErrorCode HiAISopBackend::onResizeEnd() {
    ErrorCode ret = CPUBackend::onResizeEnd();
    // alloc workspace
    mWorkspaceBuffer = SingleOpBuffer_Create(mWorkspaceSize);
    void* workspace = SingleOpBuffer_GetData(mWorkspaceBuffer);
    // Init each exec
    for (auto exec : mExecContainer) {
        if (exec->onInit(workspace) != NO_ERROR) {
            MNN_ERROR("Init op %s failed\n", (exec->name()).c_str());
            return NOT_SUPPORT;
        }
    }
    return NO_ERROR;
}

HiAI_SingleOpTensorDesc* HiAISopBackend::getHiAISopTensorDesc(const Tensor* t) {
    return mTensorManager->getHiAISopTensorDesc(t);
}

HiAI_SingleOpTensor* HiAISopBackend::getHiAISopTensor(const Tensor* t) {
    return mTensorManager->getHiAISopTensor(t);
}

bool HiAISopBackend::onUpdateTensorMem(Tensor* t) {
    return mTensorManager->onUpdateTensorMem(t);
}

void HiAISopBackend::updateWorkspaceSize(size_t execWorkspaceSize) {
    if (execWorkspaceSize > mWorkspaceSize) {
        mWorkspaceSize = execWorkspaceSize;
    }
}

bool HiAISopBackend::addHiAISopCreator(OpType t, HiAISopCreator* ct) {
    MNN_PRINT("------ add creator for [%s]\n", MNN::EnumNameOpType(t));
    auto creatorContainer = getHiAISopCreatorContainer();
    if (creatorContainer->find(t) == creatorContainer->end()) {
        creatorContainer->insert(std::make_pair(t, ct));
    }
    return true;
}

void HiAISopBackend::addSopExecution(const Execution* exec,
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // 记录单算子输入输出Tensor，后续onAcquire阶段分配ION内存
    recordHiAISopUseCount(inputs, true);
    recordHiAISopUseCount(outputs, false);
    mExecContainer.emplace_back(static_cast<HiAISopExecution*>(const_cast<Execution*>(exec)));
}

void HiAISopBackend::recordHiAISopUseCount(const std::vector<Tensor*>& tensors, bool isInput) {
    for (const auto& t : tensors) {
        if (mTensorHiAISopUseCount.find(t) == mTensorHiAISopUseCount.end()) {
            mTensorHiAISopUseCount[t] = 0;
        }
        mTensorHiAISopUseCount[t] += isInput ? -1 : 1;
    }
}

HiAISopBackend::HiAISopStorageType HiAISopBackend::getHiAISopStorageType(const Tensor* t) {
    auto iter = mTensorHiAISopUseCount.find(t);
    if (iter == mTensorHiAISopUseCount.end()) {
        return CPU_USE_ONLY;
    }
    if (iter->second == 0) {
        return HIAI_SOP_USE_ONLY;
    }
    return MIXED_USE;
}

void registerHiAISopRuntimeCreator()
{
    loadHiAISingleOpSymbol();
    registerHiAISopOps();
}
}