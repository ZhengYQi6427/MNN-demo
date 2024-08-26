#ifndef HiAISopBackend_hpp
#define HiAISopBackend_hpp

#include <vector>
#include <memory>
#include <map>
#include <optional>
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
typedef struct HiAI_SingleOpTensorDesc HiAI_SingleOpTensorDesc;
typedef struct HiAI_SingleOpTensor HiAI_SingleOpTensor;
typedef struct HiAI_SingleOpBuffer HiAI_SingleOpBuffer;
typedef struct HiAISopTensorManager HiAISopTensorManager;
typedef struct HiAISopExecution HiAISopExecution;

class HiAISopBackend : public CPUBackend {
public:
    virtual ~HiAISopBackend();
    HiAISopBackend(const CPURuntime* runtime, BackendConfig::MemoryMode memory);
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual Backend::MemObj* onAcquire(const Tensor* nativeTensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) override;
    virtual void onResizeBegin() override;
    virtual ErrorCode onResizeEnd() override;

    HiAI_SingleOpTensorDesc* getHiAISopTensorDesc(const Tensor* t);
    HiAI_SingleOpTensor* getHiAISopTensor(const Tensor* t);
    bool onUpdateTensorMem(Tensor* t);

    void updateWorkspaceSize(size_t execWorkspaceSize);
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