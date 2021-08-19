#ifdef __cplusplus
#include <map>
#include <unordered_map>
#include <iostream>
extern "C"{
#endif
#include <CL/cl.h>
#ifdef __cplusplus
}
#endif

#define DEVICE_INFO_NEED_SIZE(INFO_NAME, INFO_TYPE, DEVICE_ID_PTR, WHICH_DEVICE, \
DEVICE_INFO_SIZE, DEVICE_INFO_PTR, ALL_INFO, INFO_PTR_WITH_SIZE) \
if (#INFO_TYPE == "size_t*"){ \
} \
else{ \
}

#define GET_DEVICE_INFO(INFO_NAME, INFO_TYPE, DEVICE_ID_PTR, WHICH_DEVICE, DEVICE_INFO_SIZE, DEVICE_INFO_PTR) \
clGetDeviceInfo(*(DEVICE_ID_PTR + WHICH_DEVICE), INFO_NAME, NULL, nullptr, &DEVICE_INFO_SIZE); \
DEVICE_INFO_PTR = (cl_device_info*)malloc(DEVICE_INFO_SIZE); \
clGetDeviceInfo(*(DEVICE_ID_PTR + WHICH_DEVICE), INFO_NAME, DEVICE_INFO_SIZE, DEVICE_INFO_PTR, NULL);

#define DEVICE_INFO(INFO_NAME, INFO_TYPE, TO_STRING_METHOD, DEVICE_ID_PTR, WHICH_DEVICE, \
DEVICE_INFO_SIZE, DEVICE_INFO_PTR, ALL_INFO) \
GET_DEVICE_INFO(#INFO_NAME, INFO_TYPE, DEVICE_ID_PTR, WHICH_DEVICE, DEVICE_INFO_SIZE, DEVICE_INFO_PTR) \
if (#INFO_TYPE == "cl_device_fp_config"){ \
    if (#INFO_NAME == "CL_DEVICE_EXECUTION_CAPABILITIES"){ \
        ALL_INFO += std::string("       ") + #INFO_NAME + "(" + #INFO_TYPE + ")" + ":\n" + "         " + "error: need decode" + "\n"; \
    }\
    else{ \
        ALL_INFO += std::string("       ") + #INFO_NAME + "(" + #INFO_TYPE + ")" + ":\n" + "         " + "error: need complete" + "\n"; \
    } \
} \
else if (#INFO_TYPE == "cl_device_exec_capabilities"){ \
    ALL_INFO += std::string("       ") + #INFO_NAME + "(" + #INFO_TYPE + ")" + ":\n" + "         " + "error: need decode" + "\n"; \
} \
else if (#INFO_TYPE == "cl_device_mem_cache_type"){ \
    ALL_INFO += std::string("       ") + #INFO_NAME + "(" + #INFO_TYPE + ")" + ":\n" + "         " + "error: need decode" + "\n"; \
} \
else if (#INFO_TYPE == "cl_device_local_mem_type"){ \
    ALL_INFO += std::string("       ") + #INFO_NAME + "(" + #INFO_TYPE + ")" + ":\n" + "         " + "error: need decode" + "\n"; \
} \
else if (#INFO_TYPE == "cl_device_partition_property*"){ \
    ALL_INFO += std::string("       ") + #INFO_NAME + "(" + #INFO_TYPE + ")" + ":\n" + "         " + "error: need decode" + "\n"; \
} \
else if (#INFO_TYPE == "cl_device_affinity_domain"){ \
    ALL_INFO += std::string("       ") + #INFO_NAME + "(" + #INFO_TYPE + ")" + ":\n" + "         " + "error: need decode" + "\n"; \
} \
else if (#INFO_TYPE == "cl_device_affinity_domain"){ \
    ALL_INFO += std::string("       ") + #INFO_NAME + "(" + #INFO_TYPE + ")" + ":\n" + "         " + "error: need decode" + "\n"; \
} \
else if (#INFO_TYPE == "cl_command_queue_properties"){ \
    ALL_INFO += std::string("       ") + #INFO_NAME + "(" + #INFO_TYPE + ")" + ":\n" + "         " + "error: need decode" + "\n"; \
} \
else if (#INFO_TYPE == "cl_device_type"){ \
    ALL_INFO += std::string("       ") + #INFO_NAME + "(" + #INFO_TYPE + ")" + ":\n" + "         " + "error: need decode" + "\n"; \
} \
else if (#INFO_TYPE == "size_t*"){ \
    ALL_INFO += std::string("       ") + #INFO_NAME + "(" + #INFO_TYPE + ")" + ":\n" + "         " + "error: USE THE DEVICE_INFO_WITH_SIZE" + "\n"; \
} \
else{ \
    ALL_INFO += std::string("       ") + #INFO_NAME + "(" + #INFO_TYPE + ")" + ":\n" + "         " + /*TO_STRING_METHOD(*(INFO_TYPE*)DEVICE_INFO_PTR)*/ + "\n"; \
} \
delete DEVICE_INFO_PTR;

#define TO_STRING_DEVICE_INFO(INFO_NAME, INFO_TYPE, DEVICE_ID_PTR, WHICH_DEVICE, \
DEVICE_INFO_SIZE, DEVICE_INFO_PTR, ALL_INFO) \
DEVICE_INFO(INFO_NAME, INFO_TYPE, std::to_string, DEVICE_ID_PTR, WHICH_DEVICE, \
DEVICE_INFO_SIZE, DEVICE_INFO_PTR, ALL_INFO)

#define STRING_DEVICE_INFO(INFO_NAME, INFO_TYPE, DEVICE_ID_PTR, WHICH_DEVICE, \
DEVICE_INFO_SIZE, DEVICE_INFO_PTR, ALL_INFO) \
DEVICE_INFO(INFO_NAME, INFO_TYPE, std::string, DEVICE_ID_PTR, WHICH_DEVICE, \
DEVICE_INFO_SIZE, DEVICE_INFO_PTR, ALL_INFO)

int main(int argc, int* argv[]){
    cl_uint pnum;
    clGetPlatformIDs(NULL, nullptr, &pnum);
    cl_platform_id* platform_id_ptr = (cl_platform_id*)malloc(pnum * sizeof(cl_platform_id));
    clGetPlatformIDs(pnum, platform_id_ptr, nullptr);

    std::unordered_map<cl_platform_info, std::string> platform_info_names = { \
    {CL_PLATFORM_PROFILE, "platform_profile"}, \
    {CL_PLATFORM_VERSION, "platform_version"}, \
    {CL_PLATFORM_NAME, "platform_name"}, \
    {CL_PLATFORM_VENDOR, "platform_vendor"}, \
    {CL_PLATFORM_EXTENSIONS, "platform_extensions"}};

    cl_uint dnum = 0;
    cl_device_id* device_id_ptr = nullptr;

    size_t platform_info_size;
    cl_platform_info* platform_info;
    size_t device_info_size;
    cl_device_info* device_info;
    std::string info;
    for(int p = 0; p < pnum; ++p){
        info += "Platform " + std::to_string(p) + "\n";
        for(auto platform_info_item = platform_info_names.begin(); platform_info_item != platform_info_names.end(); ++platform_info_item){
            clGetPlatformInfo(*(platform_id_ptr + p), platform_info_item->first, NULL, nullptr, &platform_info_size);
            platform_info = (cl_platform_info*)malloc(platform_info_size);
            clGetPlatformInfo(*(platform_id_ptr + p), platform_info_item->first, platform_info_size, platform_info, NULL);
            info += "   " + platform_info_item->second + ":\n";
            info += "       " + std::string((char*)platform_info) + "\n";
            delete platform_info;
        }
        clGetDeviceIDs(*(platform_id_ptr + p), CL_DEVICE_TYPE_ALL, NULL, nullptr, &dnum);
        cl_device_id* device_id_ptr = (cl_device_id*)malloc(dnum * sizeof(cl_device_id));
        clGetDeviceIDs(*(platform_id_ptr + p), CL_DEVICE_TYPE_ALL, dnum, device_id_ptr, NULL);
        for(int d = 0; d < dnum; ++d){
            info += std::string("   |\n    ->") + "Device " + std::to_string(d) + "\n";
            TO_STRING_DEVICE_INFO(CL_DEVICE_ADDRESS_BITS, cl_uint, device_id_ptr, d, device_info_size, device_info, info);
            TO_STRING_DEVICE_INFO(CL_DEVICE_AVAILABLE, cl_bool, device_id_ptr, d, device_info_size, device_info, info);
            STRING_DEVICE_INFO(CL_DEVICE_BUILT_IN_KERNELS, char*, device_id_ptr, d, device_info_size, device_info, info);
            TO_STRING_DEVICE_INFO(CL_DEVICE_COMPILER_AVAILABLE, cl_bool, device_id_ptr, d, device_info_size, device_info, info);
            TO_STRING_DEVICE_INFO(CL_DEVICE_DOUBLE_FP_CONFIG, cl_device_fp_config, device_id_ptr, d, device_info_size, device_info, info);
            TO_STRING_DEVICE_INFO(CL_DEVICE_ENDIAN_LITTLE, cl_bool, device_id_ptr, d, device_info_size, device_info, info);
            TO_STRING_DEVICE_INFO(CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool, device_id_ptr, d, device_info_size, device_info, info);
            TO_STRING_DEVICE_INFO(CL_DEVICE_EXECUTION_CAPABILITIES, cl_device_exec_capabilities, device_id_ptr, d, device_info_size, device_info, info);
            STRING_DEVICE_INFO(CL_DEVICE_EXTENSIONS, char*, device_id_ptr, d, device_info_size, device_info, info);
            TO_STRING_DEVICE_INFO(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong, device_id_ptr, d, device_info_size, device_info, info);
            TO_STRING_DEVICE_INFO(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, cl_device_mem_cache_type, device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
            //DEVICE_INFO(, , device_id_ptr, d, device_info_size, device_info, info);
        }
        delete device_id_ptr;
    }

    delete platform_id_ptr;
    std::cout << info << std::endl;
    return 0;
}