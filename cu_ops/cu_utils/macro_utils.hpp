#ifndef PROJECT_BASE_CU_OPS_CU_UTILS_MACRO_UTILS_H
#define PROJECT_BASE_CU_OPS_CU_UTILS_MACRO_UTILS_H
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr) \
    { \
        cudaError_t error_code = callstr; \
        if (error_code != cudaSuccess) { \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0); \
        } \
    }
#endif  // CUDA_CHECK
#endif