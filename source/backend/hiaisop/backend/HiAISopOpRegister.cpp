namespace MNN {
extern void ___OpType_Convolution__HiAISopConvolutionCreator__();
extern void ___OpType_Deconvolution__HiAISopConvolutionCreator__();
extern void ___OpType_ConvolutionDepthwise__HiAISopConvolutionCreator__();
extern void ___OpType_DeconvolutionDepthwise__HiAISopConvolutionCreator__();

void registerHiAISopOps() {
#if defined(__ANDROID__) || defined(__aarch64__)
___OpType_Convolution__HiAISopConvolutionCreator__();
___OpType_Deconvolution__HiAISopConvolutionCreator__();
___OpType_ConvolutionDepthwise__HiAISopConvolutionCreator__();
___OpType_DeconvolutionDepthwise__HiAISopConvolutionCreator__();
#endif
}
}