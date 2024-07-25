#ifndef HiAISopBackend_hpp
#define HiAISopBackend_hpp

#include <vector>
#include <memory>
#include <map>
#include <optional>
#include "backend/cpu/CPUBackend.hpp"
#include "HiAISopSymbol.hpp"
#include "HiAISopTensorManager.hpp"

namespace MNN {
class HiAISopExecution : public Execution {
public:
    HiAISopExecution(Backend* backend, std::string opName) : Execution(backend) {
        mOpName = opName;
    }
    virtual ~HiAISopExecution() {
        SingleOpExecutor_Destroy(&mExec);
    }
    ErrorCode onInit(void* workspace) {
        if (SingleOpExecutor_Init(mExec, workspace, workspaceSize()) != 0) {
            return NOT_SUPPORT;
        }
        return NO_ERROR;
    }
    std::string name() const {
        return mOpName;
    }
    size_t workspaceSize() const {
        return SingleOpExecutor_GetWorkspaceSize(mExec);
    }
    HiAI_SingleOpExecutor* executor() const {
        return mExec;
    }
protected:
    void setExecutor(HiAI_SingleOpExecutor* executor) {
        if (mExec != nullptr) {
            SingleOpExecutor_Destroy(&mExec);
        }
        mExec = executor;
    }
private:
    std::string mOpName;
    HiAI_SingleOpExecutor* mExec {nullptr};
};

class HiAISopBackend : public CPUBackend {
public:
    virtual ~HiAISopBackend();
    HiAISopBackend(const CPURuntime* runtime, BackendConfig::MemoryMode memory);
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual Backend::MemObj* onAcquire(const Tensor* nativeTensor, StorageType storageType) override;
    
    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;

    HiAI_SingleOpTensorDesc* getHiAISopTensorDesc(const Tensor* t);
    HiAI_SingleOpTensor* getHiAISopTensor(const Tensor* t);
    bool onUpdateTensorMem(Tensor* t);

    void updateWorkspaceSize(size_t execWorkspaceSize);
    void* getWorkspace();
public:
    class HiAISopCreator {
    public:
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };

    static bool addHiAISopCreator(OpType t, HiAISopCreator* ct);

protected:
private:
    enum HiAISopStorageType {
        HIAI_SOP_USE_ONLY = 0,
        MIXED_USE = 1,
        CPU_USE_ONLY = 2,
    };

    size_t mWorkspaceSize = 0;
    HiAI_SingleOpBuffer* mWorkspaceBuffer;
    std::shared_ptr<HiAISopTensorManager> mTensorManager;
    std::map<const Tensor*, uint32_t> mTensorHiAISopUseCount;
    std::vector<HiAISopExecution*> mExecContainer;

    void addSopExecution(const Execution* exec, const std::vector<Tensor*>& inputs,
        const std::vector<Tensor*>& outputs);
    void recordHiAISopUseCount(const std::vector<Tensor*>& tensors, bool isInput);
    HiAISopStorageType getHiAISopStorageType(const Tensor* t);
};

#define REGISTER_HIAISOP_OP_CREATOR(type, creator) \
    void ___##type##__##creator##__() { \
        HiAISopBackend::addHiAISopCreator(type, new creator); \
    }

} // namespace MNN
#endif