#include <vector>
#include "../yolov5.hh"
int main(int* argc, char* argv[]){
    int classCount = 10, netWidth = 256, netHeight = 256, maxOut = 100;
    bool is_segmentation = false;
    std::vector<YoloKernel> vYoloKernel {
        {32, 32, {0.1, 0.1, 0.1, 0.1, 0.1, 0.1}},
        {16, 16, {0.1, 0.1, 0.1, 0.1, 0.1, 0.1}},
        {8, 8, {0.1, 0.1, 0.1, 0.1, 0.1, 0.1}},
    };
    auto p = nvinfer1::YoloLayerPlugin(classCount, netWidth, netHeight, maxOut, is_segmentation, vYoloKernel);
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    float* inputs, * output;
    unsigned long long input_size = 0;
    for(auto yk = vYoloKernel.begin(); yk != vYoloKernel.end(); ++yk){
        input_size += yk->width * yk->width * (kNumClass + 1 + kNumAnchor * 2);
    }
    CUDA_CHECK(cudaMalloc((void**)&inputs, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&output, maxOut * sizeof(Detection)));
    int batchSize = 1;
    p.forwardGpu(&inputs, output, stream, batchSize);
    //CUDA_CHECK(cudaFree(inputs));
    //CUDA_CHECK(cudaFree(output));
    return 0;
}