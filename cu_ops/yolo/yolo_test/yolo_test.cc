#include <vector>
#include "../yolov5.hh"
int main(int* argc, char* argv[]){
    int classCount = 10, netWidth = 264, netHeight = 264, maxOut = 100;
    bool is_segmentation = false;
    std::vector<YoloKernel> vYoloKernel {};
    auto p = nvinfer1::YoloLayerPlugin::YoloLayerPlugin(classCount, netWidth, netHeight, maxOut, is_segmentation, vYoloKernel);
    return 0;
}