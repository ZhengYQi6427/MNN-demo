namespace MNN {
extern void ___OpType_Convolution__HiAISopConvolutionCreator__();

void registerHiAISopOps() {
#if defined(__ANDROID__) || defined(__aarch64__)
___OpType_Convolution__HiAISopConvolutionCreator__();
#endif
}
}