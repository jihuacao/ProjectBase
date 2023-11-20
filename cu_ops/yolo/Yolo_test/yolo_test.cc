#include <vector>
#include "../yolov5.hh"
int main(int* argc, char* argv[]){
    int classCount = 10, netWidth = 256, netHeight = 256, maxOut = 100;
    bool is_segmentation = false;
    std::vector<YoloKernel> vYoloKernel {
        {16, 16, {0.1, 0.1, 0.1, 0.1, 0.1, 0.1}},
        {8, 8, {0.1, 0.1, 0.1, 0.1, 0.1, 0.1}},
        {4, 4, {0.1, 0.1, 0.1, 0.1, 0.1, 0.1}},
    };
    nvinfer1::YoloLayerPlugin* p_ptr = new nvinfer1::YoloLayerPlugin(classCount, netWidth, netHeight, maxOut, is_segmentation, vYoloKernel);
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
    p_ptr->forwardGpu(&inputs, output, stream, batchSize);
    cudaStreamSynchronize(stream);
    std::cout << "test" << std::endl;
    return 0;
}