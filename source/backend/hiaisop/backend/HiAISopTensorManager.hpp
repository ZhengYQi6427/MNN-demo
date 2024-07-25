#ifndef HiAISopTensorManager_hpp
#define HiAISopTensorManager_hpp

#include "core/Backend.hpp"
#include "HiAISopSymbol.hpp"
#include <MNN/Tensor.hpp>

namespace MNN {
#define ALIGNED_SIZE(size) (((size) + (64) - 1) / (32)) * 32

class HiAISopMemObj : public Backend::MemObj {
public:
    HiAISopMemObj(Tensor* nativeTensor, bool isVirtual);
    virtual ~HiAISopMemObj();
    inline HiAI_SingleOpTensorDesc* sopTensorDesc() {
        return mTensorDesc;
    }
    inline HiAI_SingleOpTensor* sopTensor() {
        return mTensor;
    }
    virtual MemChunk chunk() override {
        return mPoint;
    }
    inline size_t size() {
        return mSize;
    }
    bool reallocate();
private:
    MemChunk mPoint;
    size_t mSize;
    HiAI_SingleOpTensorDesc* mTensorDesc;
    HiAI_SingleOpTensor* mTensor;
};

class HiAISopTensorManager {
public:
    HiAISopTensorManager() = default;
    ~HiAISopTensorManager();
    Backend::MemObj* onAlloc(Tensor* t, bool isVirtual);
    void onRelease(Tensor* t);
    bool onUpdateTensorMem(Tensor* t);

    HiAI_SingleOpTensorDesc* getHiAISopTensorDesc(const Tensor* t);
    HiAI_SingleOpTensor* getHiAISopTensor(const Tensor* t);
    
private:
    std::map<const Tensor*, Backend::MemObj*> mMemObjContainer;
};
}

#endif