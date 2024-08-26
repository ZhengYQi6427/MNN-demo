#ifndef HiAISopConvolution_hpp
#define HiAISopConvolution_hpp

#include "HiAISopSymbol.hpp"
#include "HiAISopBackend.hpp"
#include "HiAISopExecution.hpp"

namespace MNN {
class HiAISopConvolution : public HiAISopExecution {
public:
    HiAISopConvolution(const Convolution2DCommon* common, Backend* backend,
        const float* originWeight, size_t originWeightSize, const float* originBias,
        size_t originBiasSize, OpType opType, std::string opName);
    virtual ~HiAISopConvolution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    HiAI_SingleOp_SupportStatus PreCheck(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

private:
    const Convolution2DCommon* mCommon;
    HiAI_SingleOp_ConvMode mMode;
    HiAI_SingleOpTensor* mWeight;
    HiAI_SingleOpTensor* mBias;
};

} // namespace MNN

#endif