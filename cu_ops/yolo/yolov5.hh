#ifndef __YOLO_V5__HH__
#define __YOLO_V5__HH__
#include <vector>
#include <string>

#ifndef __MACROS_H
#define __MACROS_H

#include <NvInfer.h>

#ifdef API_EXPORTS
#if defined(_MSC_VER)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
#else

#if defined(_MSC_VER)
#define API __declspec(dllimport)
#else
#define API
#endif
#endif  // API_EXPORTS

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif

#endif  // __MACROS_H

// For INT8, you need prepare the calibration dataset, please refer to
// https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5#int8-quantization
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32

// These are used to define input/output tensor names,
// you can set them to whatever you want.
const static char* kInputTensorName = "data";
const static char* kOutputTensorName = "prob";

// Detection model and Segmentation model' number of classes
constexpr static int kNumClass = 80;

// Classfication model's number of classes
constexpr static int kClsNumClass = 1000;

constexpr static int kBatchSize = 1;

// Yolo's input width and height must by divisible by 32
constexpr static int kInputH = 640;
constexpr static int kInputW = 640;

// Classfication model's input shape
constexpr static int kClsInputH = 224;
constexpr static int kClsInputW = 224;

// Maximum number of output bounding boxes from yololayer plugin.
// That is maximum number of output bounding boxes before NMS.
constexpr static int kMaxNumOutputBbox = 1000;

constexpr static int kNumAnchor = 3;

// The bboxes whose confidence is lower than kIgnoreThresh will be ignored in yololayer plugin.
constexpr static float kIgnoreThresh = 0.1f;

/* --------------------------------------------------------
 * These configs are NOT related to tensorrt model, if these are changed,
 * please re-compile, but no need to re-serialize the tensorrt model.
 * --------------------------------------------------------*/

// NMS overlapping thresh and final detection confidence thresh
const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.5f;

const static int kGpuId = 0;

// If your image size is larger than 4096 * 3112, please increase this value
const static int kMaxInputImageSize = 4096 * 3112;

struct YoloKernel {
  int width;
  int height;
  float anchors[kNumAnchor * 2];
};

struct alignas(float) Detection {
  float bbox[4];  // center_x center_y w h
  float conf;  // bbox_conf * cls_conf
  float class_id;
  float mask[32];
};

namespace nvinfer1 {
class API YoloLayerPlugin : public IPluginV2IOExt {
public:
  YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, bool is_segmentation, const std::vector<YoloKernel>& vYoloKernel);
  YoloLayerPlugin(const void* data, size_t length);
  ~YoloLayerPlugin();

  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT override;

  int initialize() TRT_NOEXCEPT override;

  virtual void terminate() TRT_NOEXCEPT override {};

  virtual size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override { return 0; }

  virtual int enqueue(int batchSize, const void* const* inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

  virtual size_t getSerializationSize() const TRT_NOEXCEPT override;

  virtual void serialize(void* buffer) const TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT override {
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
  }

  const char* getPluginType() const TRT_NOEXCEPT override;

  const char* getPluginVersion() const TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override;

  IPluginV2IOExt* clone() const TRT_NOEXCEPT override;

  void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override;

  const char* getPluginNamespace() const TRT_NOEXCEPT override;

  DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT override;

  bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT override;

  bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override;

  void attachToContext(
      cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT override;

  void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT override;

  void detachFromContext() TRT_NOEXCEPT override;

 private:
  void forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize = 1);
  int mThreadCount = 256;
  const char* mPluginNamespace;
  int mKernelCount;
  int mClassCount;
  int mYoloV5NetWidth;
  int mYoloV5NetHeight;
  int mMaxOutObject;
  bool is_segmentation_;
  std::vector<YoloKernel> mYoloKernel;
  void** mAnchor;
};

class API YoloPluginCreator : public IPluginCreator {
 public:
  YoloPluginCreator();

  ~YoloPluginCreator() override = default;

  const char* getPluginName() const TRT_NOEXCEPT override;

  const char* getPluginVersion() const TRT_NOEXCEPT override;

  const PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;

  IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT override;

  IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT override;

  void setPluginNamespace(const char* libNamespace) TRT_NOEXCEPT override {
    mNamespace = libNamespace;
  }

  const char* getPluginNamespace() const TRT_NOEXCEPT override {
    return mNamespace.c_str();
  }

 private:
  std::string mNamespace;
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
};
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};
#endif