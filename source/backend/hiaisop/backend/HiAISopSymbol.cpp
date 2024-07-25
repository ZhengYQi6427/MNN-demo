#include "HiAISopSymbol.hpp"
#include <MNN/MNNDefine.h>
#include <dlfcn.h>

namespace MNN {

#define LOAD_SYM(NAME)                                                                   \
    NAME = reinterpret_cast<decltype(NAME)>(dlsym(lib, "HiAI_" #NAME)); \
    if (NAME == nullptr) {                                                        \
        MNN_PRINT("[NNAPI] Load symbol %s failed.", "HiAI_" #NAME);                                         \
        return false;                                                                               \
    }

bool loadHiAISingleOpSymbol() {
    void *lib = dlopen("/vendor/lib64/libai_single_op.so", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        return false;
    }
    LOAD_SYM(SingleOpTensorDesc_Create);
    LOAD_SYM(SingleOpTensorDesc_GetDimensionCount);
    LOAD_SYM(SingleOpTensorDesc_GetDimension);
    LOAD_SYM(SingleOpTensorDesc_GetDataType);
    LOAD_SYM(SingleOpTensorDesc_GetFormat);
    LOAD_SYM(SingleOpTensorDesc_IsVirtual);
    LOAD_SYM(SingleOpTensorDesc_GetByteSize);
    LOAD_SYM(SingleOpTensorDesc_Destroy);
    LOAD_SYM(SingleOpBuffer_Create);
    LOAD_SYM(SingleOpBuffer_GetSize);
    LOAD_SYM(SingleOpBuffer_GetData);
    LOAD_SYM(SingleOpBuffer_Destroy);
    LOAD_SYM(SingleOpTensor_CreateFromTensorDesc);
    LOAD_SYM(SingleOpTensor_CreateFromSingleOpBuffer);
    LOAD_SYM(SingleOpTensor_CreateFromConst);
    LOAD_SYM(SingleOpTensor_GetTensorDesc);
    LOAD_SYM(SingleOpTensor_GetBuffer);
    LOAD_SYM(SingleOpTensor_Destroy);
    LOAD_SYM(SingleOpOptions_Create);
    LOAD_SYM(SingleOpOptions_Destroy);
    LOAD_SYM(SingleOpDescriptor_CreateConvolution);
    LOAD_SYM(SingleOpDescriptor_CreateActivation);
    LOAD_SYM(SingleOpDescriptor_Destroy);
    LOAD_SYM(SingleOpExecutor_PreCheckConvolution);
    LOAD_SYM(SingleOpExecutor_CreateConvolution);
    LOAD_SYM(SingleOpExecutor_PreCheckFusedConvolutionActivation);
    LOAD_SYM(SingleOpExecutor_CreateFusedConvolutionActivation);
    LOAD_SYM(SingleOpExecutor_Destroy);
    LOAD_SYM(SingleOpExecutor_UpdateOutputTensorDesc);
    LOAD_SYM(SingleOpExecutor_GetWorkspaceSize);
    LOAD_SYM(SingleOpExecutor_Init);
    LOAD_SYM(SingleOpExecutor_Execute);

    return true;
}

Func_HiAI_SingleOpTensorDesc_Create SingleOpTensorDesc_Create;
Func_HiAI_SingleOpTensorDesc_GetDimensionCount SingleOpTensorDesc_GetDimensionCount;
Func_HiAI_SingleOpTensorDesc_GetDimension SingleOpTensorDesc_GetDimension;
Func_HiAI_SingleOpTensorDesc_GetDataType SingleOpTensorDesc_GetDataType;
Func_HiAI_SingleOpTensorDesc_GetFormat SingleOpTensorDesc_GetFormat;
Func_HiAI_SingleOpTensorDesc_IsVirtual SingleOpTensorDesc_IsVirtual;
Func_HiAI_SingleOpTensorDesc_GetByteSize SingleOpTensorDesc_GetByteSize;
Func_HiAI_SingleOpTensorDesc_Destroy SingleOpTensorDesc_Destroy;
Func_HiAI_SingleOpBuffer_Create SingleOpBuffer_Create;
Func_HiAI_SingleOpBuffer_GetSize SingleOpBuffer_GetSize;
Func_HiAI_SingleOpBuffer_GetData SingleOpBuffer_GetData;
Func_HiAI_SingleOpBuffer_Destroy SingleOpBuffer_Destroy;
Func_HiAI_SingleOpTensor_CreateFromTensorDesc SingleOpTensor_CreateFromTensorDesc;
Func_HiAI_SingleOpTensor_CreateFromSingleOpBuffer SingleOpTensor_CreateFromSingleOpBuffer;
Func_HiAI_SingleOpTensor_CreateFromConst SingleOpTensor_CreateFromConst;
Func_HiAI_SingleOpTensor_GetTensorDesc SingleOpTensor_GetTensorDesc;
Func_HiAI_SingleOpTensor_GetBuffer SingleOpTensor_GetBuffer;
Func_HiAI_SingleOpTensor_Destroy SingleOpTensor_Destroy;
Func_HiAI_SingleOpOptions_Create SingleOpOptions_Create;
Func_HiAI_SingleOpOptions_Destroy SingleOpOptions_Destroy;
Func_HiAI_SingleOpDescriptor_CreateConvolution SingleOpDescriptor_CreateConvolution;
Func_HiAI_SingleOpDescriptor_CreateActivation SingleOpDescriptor_CreateActivation;
Func_HiAI_SingleOpDescriptor_Destroy SingleOpDescriptor_Destroy;
Func_HiAI_SingleOpExecutor_PreCheckConvolution SingleOpExecutor_PreCheckConvolution;
Func_HiAI_SingleOpExecutor_CreateConvolution SingleOpExecutor_CreateConvolution;
Func_HiAI_SingleOpExecutor_PreCheckFusedConvolutionActivation SingleOpExecutor_PreCheckFusedConvolutionActivation;
Func_HiAI_SingleOpExecutor_CreateFusedConvolutionActivation SingleOpExecutor_CreateFusedConvolutionActivation;
Func_HiAI_SingleOpExecutor_Destroy SingleOpExecutor_Destroy;
Func_HiAI_SingleOpExecutor_UpdateOutputTensorDesc SingleOpExecutor_UpdateOutputTensorDesc;
Func_HiAI_SingleOpExecutor_GetWorkspaceSize SingleOpExecutor_GetWorkspaceSize;
Func_HiAI_SingleOpExecutor_Init SingleOpExecutor_Init;
Func_HiAI_SingleOpExecutor_Execute SingleOpExecutor_Execute;
}