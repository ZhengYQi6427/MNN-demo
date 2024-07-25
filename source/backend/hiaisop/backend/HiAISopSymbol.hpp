#ifndef HIAI_SOP_SYMBOL_HPP
#define HIAI_SOP_SYMBOL_HPP

#include "single_op/single_op_all.h"

namespace MNN {
typedef HiAI_SingleOpTensorDesc* (*Func_HiAI_SingleOpTensorDesc_Create)(const int64_t* dims, size_t dimNum,
    HiAI_SingleOp_DataType dataType, HiAI_SingleOp_Format format, bool isVirtual);
typedef size_t (*Func_HiAI_SingleOpTensorDesc_GetDimensionCount)(const HiAI_SingleOpTensorDesc* tensorDesc);
typedef int64_t (*Func_HiAI_SingleOpTensorDesc_GetDimension)(const HiAI_SingleOpTensorDesc* tensorDesc, size_t index);
typedef HiAI_SingleOp_DataType (*Func_HiAI_SingleOpTensorDesc_GetDataType)(const HiAI_SingleOpTensorDesc* tensorDesc);
typedef HiAI_SingleOp_Format (*Func_HiAI_SingleOpTensorDesc_GetFormat)(const HiAI_SingleOpTensorDesc* tensorDesc);
typedef bool (*Func_HiAI_SingleOpTensorDesc_IsVirtual)(const HiAI_SingleOpTensorDesc* tensorDesc);
typedef size_t (*Func_HiAI_SingleOpTensorDesc_GetByteSize)(const HiAI_SingleOpTensorDesc* tensorDesc);
typedef void (*Func_HiAI_SingleOpTensorDesc_Destroy)(HiAI_SingleOpTensorDesc** tensorDesc);

typedef HiAI_SingleOpBuffer* (*Func_HiAI_SingleOpBuffer_Create)(size_t dataSize);
typedef size_t (*Func_HiAI_SingleOpBuffer_GetSize)(const HiAI_SingleOpBuffer* buffer);
typedef void* (*Func_HiAI_SingleOpBuffer_GetData)(const HiAI_SingleOpBuffer* buffer);
typedef HIAI_Status (*Func_HiAI_SingleOpBuffer_Destroy)(HiAI_SingleOpBuffer** buffer);

typedef HiAI_SingleOpTensor* (*Func_HiAI_SingleOpTensor_CreateFromTensorDesc)(const HiAI_SingleOpTensorDesc* desc);
typedef HiAI_SingleOpTensor* (*Func_HiAI_SingleOpTensor_CreateFromSingleOpBuffer)(const HiAI_SingleOpTensorDesc* desc,
    void* data, size_t dataSize);
typedef HiAI_SingleOpTensor* (*Func_HiAI_SingleOpTensor_CreateFromConst)(const HiAI_SingleOpTensorDesc* desc,
    void* data, size_t dataSize);
typedef HiAI_SingleOpTensorDesc* (*Func_HiAI_SingleOpTensor_GetTensorDesc)(const HiAI_SingleOpTensor* tensor);
typedef HiAI_SingleOpBuffer* (*Func_HiAI_SingleOpTensor_GetBuffer)(const HiAI_SingleOpTensor* tensor);
typedef HIAI_Status (*Func_HiAI_SingleOpTensor_Destroy)(HiAI_SingleOpTensor** tensor);

typedef HiAI_SingleOpOptions* (*Func_HiAI_SingleOpOptions_Create)(void);
typedef void (*Func_HiAI_SingleOpOptions_Destroy)(HiAI_SingleOpOptions** options);
typedef HiAI_SingleOpDescriptor* (*Func_HiAI_SingleOpDescriptor_CreateConvolution)
    (HiAI_SingleOp_ConvMode convMode, const int64_t strides[2], const int64_t dilations[2],
    const int64_t pads[4], int64_t groups, HiAI_SingleOp_PadMode padMode);
typedef HiAI_SingleOpDescriptor* (*Func_HiAI_SingleOpDescriptor_CreateActivation)
    (HiAI_SingleOp_ActivationType activationType, float coef);
typedef void (*Func_HiAI_SingleOpDescriptor_Destroy)(HiAI_SingleOpDescriptor** opDesc);
typedef HiAI_SingleOp_SupportStatus (*Func_HiAI_SingleOpExecutor_PreCheckConvolution)
    (HiAI_SingleOpOptions* options, HiAI_SingleOpDescriptor* opDesc, HiAI_SingleOpTensorDesc* input,
    HiAI_SingleOpTensorDesc* output, HiAI_SingleOpTensor* filter, HiAI_SingleOpTensor* bias);
typedef HiAI_SingleOpExecutor* (*Func_HiAI_SingleOpExecutor_CreateConvolution)
    (HiAI_SingleOpOptions* options, HiAI_SingleOpDescriptor* opDesc, HiAI_SingleOpTensorDesc* input,
    HiAI_SingleOpTensorDesc* output, HiAI_SingleOpTensor* filter, HiAI_SingleOpTensor* bias);
typedef HiAI_SingleOp_SupportStatus (*Func_HiAI_SingleOpExecutor_PreCheckFusedConvolutionActivation)
    (HiAI_SingleOpOptions* options, HiAI_SingleOpDescriptor* convOpDesc, HiAI_SingleOpDescriptor* actOpDesc,
    HiAI_SingleOpTensorDesc* input, HiAI_SingleOpTensorDesc* output, HiAI_SingleOpTensor* filter,
    HiAI_SingleOpTensor* bias);
typedef HiAI_SingleOpExecutor* (*Func_HiAI_SingleOpExecutor_CreateFusedConvolutionActivation)
    (HiAI_SingleOpOptions* options, HiAI_SingleOpDescriptor* convOpDesc, HiAI_SingleOpDescriptor* actOpDesc,
    HiAI_SingleOpTensorDesc* input, HiAI_SingleOpTensorDesc* output, HiAI_SingleOpTensor* filter,
    HiAI_SingleOpTensor* bias);
typedef HIAI_Status (*Func_HiAI_SingleOpExecutor_Destroy)(HiAI_SingleOpExecutor** executor);
typedef HIAI_Status (*Func_HiAI_SingleOpExecutor_UpdateOutputTensorDesc)(HiAI_SingleOpExecutor* executor,
    uint32_t index, HiAI_SingleOpTensorDesc* output);
typedef size_t (*Func_HiAI_SingleOpExecutor_GetWorkspaceSize)(const HiAI_SingleOpExecutor* executor);
typedef HIAI_Status (*Func_HiAI_SingleOpExecutor_Init)(HiAI_SingleOpExecutor* executor, void* workspace,
    size_t worspaceSize);
typedef HIAI_Status (*Func_HiAI_SingleOpExecutor_Execute)(HiAI_SingleOpExecutor* executor,
    HiAI_SingleOpTensor* input[], int32_t inputNum, HiAI_SingleOpTensor* output[], int32_t outputNum);

extern Func_HiAI_SingleOpTensorDesc_Create SingleOpTensorDesc_Create;
extern Func_HiAI_SingleOpTensorDesc_GetDimensionCount SingleOpTensorDesc_GetDimensionCount;
extern Func_HiAI_SingleOpTensorDesc_GetDimension SingleOpTensorDesc_GetDimension;
extern Func_HiAI_SingleOpTensorDesc_GetDataType SingleOpTensorDesc_GetDataType;
extern Func_HiAI_SingleOpTensorDesc_GetFormat SingleOpTensorDesc_GetFormat;
extern Func_HiAI_SingleOpTensorDesc_IsVirtual SingleOpTensorDesc_IsVirtual;
extern Func_HiAI_SingleOpTensorDesc_GetByteSize SingleOpTensorDesc_GetByteSize;
extern Func_HiAI_SingleOpTensorDesc_Destroy SingleOpTensorDesc_Destroy;
extern Func_HiAI_SingleOpBuffer_Create SingleOpBuffer_Create;
extern Func_HiAI_SingleOpBuffer_GetSize SingleOpBuffer_GetSize;
extern Func_HiAI_SingleOpBuffer_GetData SingleOpBuffer_GetData;
extern Func_HiAI_SingleOpBuffer_Destroy SingleOpBuffer_Destroy;
extern Func_HiAI_SingleOpTensor_CreateFromTensorDesc SingleOpTensor_CreateFromTensorDesc;
extern Func_HiAI_SingleOpTensor_CreateFromSingleOpBuffer SingleOpTensor_CreateFromSingleOpBuffer;
extern Func_HiAI_SingleOpTensor_CreateFromConst SingleOpTensor_CreateFromConst;
extern Func_HiAI_SingleOpTensor_GetTensorDesc SingleOpTensor_GetTensorDesc;
extern Func_HiAI_SingleOpTensor_GetBuffer SingleOpTensor_GetBuffer;
extern Func_HiAI_SingleOpTensor_Destroy SingleOpTensor_Destroy;
extern Func_HiAI_SingleOpOptions_Create SingleOpOptions_Create;
extern Func_HiAI_SingleOpOptions_Destroy SingleOpOptions_Destroy;
extern Func_HiAI_SingleOpDescriptor_CreateConvolution SingleOpDescriptor_CreateConvolution;
extern Func_HiAI_SingleOpDescriptor_CreateActivation SingleOpDescriptor_CreateActivation;
extern Func_HiAI_SingleOpDescriptor_Destroy SingleOpDescriptor_Destroy;
extern Func_HiAI_SingleOpExecutor_PreCheckConvolution SingleOpExecutor_PreCheckConvolution;
extern Func_HiAI_SingleOpExecutor_CreateConvolution SingleOpExecutor_CreateConvolution;
extern Func_HiAI_SingleOpExecutor_PreCheckFusedConvolutionActivation SingleOpExecutor_PreCheckFusedConvolutionActivation;
extern Func_HiAI_SingleOpExecutor_CreateFusedConvolutionActivation SingleOpExecutor_CreateFusedConvolutionActivation;
extern Func_HiAI_SingleOpExecutor_Destroy SingleOpExecutor_Destroy;
extern Func_HiAI_SingleOpExecutor_UpdateOutputTensorDesc SingleOpExecutor_UpdateOutputTensorDesc;
extern Func_HiAI_SingleOpExecutor_GetWorkspaceSize SingleOpExecutor_GetWorkspaceSize;
extern Func_HiAI_SingleOpExecutor_Init SingleOpExecutor_Init;
extern Func_HiAI_SingleOpExecutor_Execute SingleOpExecutor_Execute;

bool loadHiAISingleOpSymbol();
}
#endif