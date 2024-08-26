#ifndef HiAISopExecution_hpp
#define HiAISopExecution_hpp

#include "HiAISopSymbol.hpp"
#include "core/Execution.hpp"

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
        if (SingleOpExecutor_Init(mExec, workspace, this->workspaceSize()) != 0) {
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
    virtual HiAI_SingleOp_SupportStatus PreCheck(const std::vector<Tensor*>& inputs,
        const std::vector<Tensor*>& outputs) {
        return HIAI_SINGLEOP_OPTIMIZED;
    }
protected:
    void setExecutor(HiAI_SingleOpExecutor* executor) {
        if (mExec != nullptr) {
            SingleOpExecutor_Destroy(&mExec);
        }
        mExec = executor;
    }
private:
    std::string mOpName = "";
    HiAI_SingleOpExecutor* mExec = nullptr;
};
}
#endif