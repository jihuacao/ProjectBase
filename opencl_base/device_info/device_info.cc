#ifdef __cplusplus
#include <vector>
#include <map>
#include <unordered_map>
#include <iostream>
extern "C"{
#endif
#include <CL/cl.h>
#ifdef __cplusplus
}
#endif
#include "ProjectBase/cross_platform/util_method.hpp"

#define DEVICE_INFO_NEED_SIZE(INFO_NAME, INFO_TYPE, DEVICE_ID_PTR, WHICH_DEVICE, \
DEVICE_INFO_SIZE, DEVICE_INFO_PTR, ALL_INFO, INFO_PTR_WITH_SIZE) \
if (#INFO_TYPE == "size_t*"){ \
} \
else{ \
}

enum InfoType{
    CLUINT,
    CLBOOL,
    CHAR,
    CLDEVICEFPCONFIG,
    TheFlag,
};

struct InfoTypeInfo{
    std::string _name;
    InfoType _type;
    std::string _help;
    std::vector<std::string> _bit_field_name;
    public:
        InfoTypeInfo(const std::string& name, \
        InfoType type, \
        const std::string& help="", \
        const std::vector<std::string>& bit_field_name=std::vector<std::string>(), \
        InfoType flag=TheFlag)
        : _name(name), 
        _type(type), 
        _help(help), 
        _bit_field_name(bit_field_name)
        {};
};

#define Default_Info_Format(INFO_TYPE_INFO, TYPE) \
std::string("       ") + INFO_TYPE_INFO._name + "("  + TYPE + ")" + "[" + INFO_TYPE_INFO._help + "]" + ":\n" + "         "

#define ToStringDeviceInfoFormat(INFO_TYPE_INFO, TYPE, DEVICE_INFO_PPTR) \
Default_Info_Format(INFO_TYPE_INFO, #TYPE) + std::to_string(*(TYPE*)*DEVICE_INFO_PPTR) + "\n";

#define PtrStringDeviceInfoFormat(INFO_TYPE_INFO, TYPE, DEVICE_INFO_PPTR) \
Default_Info_Format(INFO_TYPE_INFO, #TYPE) + std::string((TYPE*)*DEVICE_INFO_PPTR) + "\n";

#define BitDeviceInfoFormat(INFO_TYPE_INFO, TYPE, DEVICE_INFO_PPTR) \
std::string bit_field_name; \
std::vector<int> char_num; \
for(auto bit_field_name_item = INFO_TYPE_INFO._bit_field_name.begin(); bit_field_name_item != INFO_TYPE_INFO._bit_field_name.end(); ++bit_field_name_item){ \
    char_num.push_back(bit_field_name_item->size()); \
    bit_field_name += std::string(" | ") + *bit_field_name_item; \
} \
std::string bit_field_flag = "         "; \
int i = 0; \
TYPE f = 1; \
for(auto char_num_item = char_num.begin(); char_num_item != char_num.end(); ++char_num_item){ \
    std::cout << *char_num_item << std::endl; \
    bit_field_flag += std::string(" | ") + std::string((((*(TYPE*)*DEVICE_INFO_PPTR) << (sizeof(TYPE) * 8 - i)) >= (f << (sizeof(TYPE) * 8 - i))) ? "1" : "0") + std::string(*char_num_item - 1, ' '); \
    ++i; \
} \
std::string temp = Default_Info_Format(INFO_TYPE_INFO, #TYPE) + bit_field_name + "\n" + bit_field_flag + "\n";

std::string get_device_info(cl_device_info info, const InfoTypeInfo info_type_info, \
cl_device_id device_id, size_t* device_info_size, cl_device_info** device_info_pptr){
    clGetDeviceInfo(device_id, info, NULL, nullptr, device_info_size);
    *device_info_pptr = (cl_device_info*)malloc(*device_info_size);
    clGetDeviceInfo(device_id, info, *device_info_size, *device_info_pptr, NULL);
    switch(info_type_info._type){
        case CLUINT:{
            return ToStringDeviceInfoFormat(info_type_info, cl_uint, device_info_pptr);
            break;
        }
        case CLBOOL:{
            return ToStringDeviceInfoFormat(info_type_info, cl_bool, device_info_pptr);
            break;
        }
        case CHAR:{
            return PtrStringDeviceInfoFormat(info_type_info, char, device_info_pptr);
            break;
        }
        case CLDEVICEFPCONFIG:{
            BitDeviceInfoFormat(info_type_info, cl_device_fp_config, device_info_pptr);
            return temp;
            break;
        }
        default:{
            return std::string("       ") + info_type_info._name + "(**)" + ":\n" + "         type not defined\n";
            break;
        }
    }
    delete *device_info_pptr;
    return "";
}

int get_info(){
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
        std::unordered_map<int, InfoTypeInfo> device_infos;
        device_infos.insert({CL_DEVICE_ADDRESS_BITS, InfoTypeInfo("CL_DEVICE_ADDRESS_BITS", CLUINT)});
        device_infos.insert({CL_DEVICE_AVAILABLE, InfoTypeInfo("CL_DEVICE_AVAILABLE", CLBOOL)});
        device_infos.insert({CL_DEVICE_BUILT_IN_KERNELS, InfoTypeInfo("CL_DEVICE_BUILT_IN_KERNELS", CHAR, "a set of supported built-in kernels")});
        device_infos.insert({CL_DEVICE_COMPILER_AVAILABLE, InfoTypeInfo("CL_DEVICE_COMPILER_AVAILABLE", CLBOOL)});
        device_infos.insert({CL_DEVICE_DOUBLE_FP_CONFIG, InfoTypeInfo("CL_DEVICE_DOUBLE_FP_CONFIG", CLDEVICEFPCONFIG, "", {"CL_FP_FMA", "CL_FP_ROUND_TO_NEAREST" ,"CL_FP_ROUND_TO_ZERO", "CL_FP_ROUND_TO_INF", "CL_FP_INF_NAN" ,"CL_FP_DENORM"})});
        //device_infos.insert({CL_DEVICE_ENDIAN_LITTLE, InfoTypeInfo("CL_DEVICE_ENDIAN_LITTLE",)});
        //device_infos.insert({CL_DEVICE_ERROR_CORRECTION_SUPPORT, InfoTypeInfo("CL_DEVICE_ERROR_CORRECTION_SUPPORT",)});
        //device_infos.insert({CL_DEVICE_EXECUTION_CAPABILITIES, InfoTypeInfo("CL_DEVICE_EXECUTION_CAPABILITIES",)});
        //device_infos.insert({CL_DEVICE_EXTENSIONS, InfoTypeInfo("CL_DEVICE_EXTENSIONS",)});
        //device_infos.insert({CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, InfoTypeInfo("CL_DEVICE_GLOBAL_MEM_CACHE_SIZE",)});
        //device_infos.insert({CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, InfoTypeInfo("CL_DEVICE_GLOBAL_MEM_CACHE_TYPE",)});
        //device_infos.insert({CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, InfoTypeInfo("CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE",)});
        //device_infos.insert({CL_DEVICE_GLOBAL_MEM_SIZE, InfoTypeInfo("CL_DEVICE_GLOBAL_MEM_SIZE",)});
        ////device_insme.insert({CL_DEVICE_HALF_FP_CONFIG, InfoTypeInfo("CL_DEVICE_HALF_FP_CONFIG",)});
        //device_infos.insert({CL_DEVICE_HOST_UNIFIED_MEMORY, InfoTypeInfo("CL_DEVICE_HOST_UNIFIED_MEMORY",)});
        //device_infos.insert({CL_DEVICE_IMAGE_SUPPORT, InfoTypeInfo("CL_DEVICE_IMAGE_SUPPORT",)});
        //device_infos.insert({CL_DEVICE_IMAGE2D_MAX_HEIGHT, InfoTypeInfo("CL_DEVICE_IMAGE2D_MAX_HEIGHT",)});
        //device_infos.insert({CL_DEVICE_IMAGE2D_MAX_WIDTH, InfoTypeInfo("CL_DEVICE_IMAGE2D_MAX_WIDTH",)});
        //device_infos.insert({CL_DEVICE_IMAGE3D_MAX_DEPTH, InfoTypeInfo("CL_DEVICE_IMAGE3D_MAX_DEPTH",)});
        //device_infos.insert({CL_DEVICE_IMAGE3D_MAX_HEIGHT, InfoTypeInfo("CL_DEVICE_IMAGE3D_MAX_HEIGHT",)});
        //device_infos.insert({CL_DEVICE_IMAGE3D_MAX_WIDTH, InfoTypeInfo("CL_DEVICE_IMAGE3D_MAX_WIDTH",)});
        //device_infos.insert({CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, InfoTypeInfo("CL_DEVICE_IMAGE_MAX_BUFFER_SIZE",)});
        //device_infos.insert({CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, InfoTypeInfo("CL_DEVICE_IMAGE_MAX_ARRAY_SIZE",)});
        //device_infos.insert({CL_DEVICE_LINKER_AVAILABLE, InfoTypeInfo("CL_DEVICE_LINKER_AVAILABLE",)});
        //device_infos.insert({CL_DEVICE_LOCAL_MEM_SIZE, InfoTypeInfo("CL_DEVICE_LOCAL_MEM_SIZE",)});
        //device_infos.insert({CL_DEVICE_LOCAL_MEM_TYPE, InfoTypeInfo("CL_DEVICE_LOCAL_MEM_TYPE",)});
        //device_infos.insert({CL_DEVICE_MAX_CLOCK_FREQUENCY, InfoTypeInfo("CL_DEVICE_MAX_CLOCK_FREQUENCY",)});
        //device_infos.insert({CL_DEVICE_MAX_COMPUTE_UNITS, InfoTypeInfo("CL_DEVICE_MAX_COMPUTE_UNITS",)});
        //device_infos.insert({CL_DEVICE_MAX_CONSTANT_ARGS, InfoTypeInfo("CL_DEVICE_MAX_CONSTANT_ARGS",)});
        //device_infos.insert({CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, InfoTypeInfo("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE",)});
        //device_infos.insert({CL_DEVICE_MAX_MEM_ALLOC_SIZE, InfoTypeInfo("CL_DEVICE_MAX_MEM_ALLOC_SIZE",)});
        //device_infos.insert({CL_DEVICE_MAX_PARAMETER_SIZE, InfoTypeInfo("CL_DEVICE_MAX_PARAMETER_SIZE",)});
        //device_infos.insert({CL_DEVICE_MAX_READ_IMAGE_ARGS, InfoTypeInfo("CL_DEVICE_MAX_READ_IMAGE_ARGS",)});
        //device_infos.insert({CL_DEVICE_MAX_SAMPLERS, InfoTypeInfo("CL_DEVICE_MAX_SAMPLERS",)});
        //device_infos.insert({CL_DEVICE_MAX_WORK_GROUP_SIZE, InfoTypeInfo("CL_DEVICE_MAX_WORK_GROUP_SIZE",)});
        //device_infos.insert({CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, InfoTypeInfo("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS",)});
        //device_infos.insert({CL_DEVICE_MAX_WORK_ITEM_SIZES, InfoTypeInfo("CL_DEVICE_MAX_WORK_ITEM_SIZES",)});
        //device_infos.insert({CL_DEVICE_MAX_WRITE_IMAGE_ARGS, InfoTypeInfo("CL_DEVICE_MAX_WRITE_IMAGE_ARGS",)});
        //device_infos.insert({CL_DEVICE_MEM_BASE_ADDR_ALIGN, InfoTypeInfo("CL_DEVICE_MEM_BASE_ADDR_ALIGN",)});
        //device_infos.insert({CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, InfoTypeInfo("CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE",)});
        //device_infos.insert({CL_DEVICE_NAME, InfoTypeInfo("CL_DEVICE_NAME",)});
        //device_infos.insert({CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR",)});
        //device_infos.insert({CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT",)});
        //device_infos.insert({CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_INT",)});
        //device_infos.insert({CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG",)});
        //device_infos.insert({CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT",)});
        //device_infos.insert({CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE",)});
        //device_infos.insert({CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF",)});
        //device_infos.insert({CL_DEVICE_OPENCL_C_VERSION, InfoTypeInfo("CL_DEVICE_OPENCL_C_VERSION",)});
        //device_infos.insert({CL_DEVICE_PARENT_DEVICE, InfoTypeInfo("CL_DEVICE_PARENT_DEVICE",)});
        //device_infos.insert({CL_DEVICE_PARTITION_MAX_SUB_DEVICES, InfoTypeInfo("CL_DEVICE_PARTITION_MAX_SUB_DEVICES",)});
        //device_infos.insert({CL_DEVICE_PARTITION_PROPERTIES, InfoTypeInfo("CL_DEVICE_PARTITION_PROPERTIES",)});
        //device_infos.insert({CL_DEVICE_PARTITION_AFFINITY_DOMAIN, InfoTypeInfo("CL_DEVICE_PARTITION_AFFINITY_DOMAIN",)});
        //device_infos.insert({CL_DEVICE_PARTITION_TYPE, InfoTypeInfo("CL_DEVICE_PARTITION_TYPE",)});
        //device_infos.insert({CL_DEVICE_PLATFORM, InfoTypeInfo("CL_DEVICE_PLATFORM",)});
        //device_infos.insert({CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR",)});
        //device_infos.insert({CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT",)});
        //device_infos.insert({CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT",)});
        //device_infos.insert({CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG",)});
        //device_infos.insert({CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT",)});
        //device_infos.insert({CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE",)});
        //device_infos.insert({CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF",)});
        //device_infos.insert({CL_DEVICE_PRINTF_BUFFER_SIZE, InfoTypeInfo("CL_DEVICE_PRINTF_BUFFER_SIZE",)});
        //device_infos.insert({CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, InfoTypeInfo("CL_DEVICE_PREFERRED_INTEROP_USER_SYNC",)});
        //device_infos.insert({CL_DEVICE_PROFILE, InfoTypeInfo("CL_DEVICE_PROFILE",)});
        //device_infos.insert({CL_DEVICE_PROFILING_TIMER_RESOLUTION, InfoTypeInfo("CL_DEVICE_PROFILING_TIMER_RESOLUTION",)});
        //device_infos.insert({CL_DEVICE_QUEUE_PROPERTIES, InfoTypeInfo("CL_DEVICE_QUEUE_PROPERTIES",)});
        //device_infos.insert({CL_DEVICE_REFERENCE_COUNT, InfoTypeInfo("CL_DEVICE_REFERENCE_COUNT",)});
        //device_infos.insert({CL_DEVICE_SINGLE_FP_CONFIG, InfoTypeInfo("CL_DEVICE_SINGLE_FP_CONFIG",)});
        //device_infos.insert({CL_DEVICE_TYPE, InfoTypeInfo("CL_DEVICE_TYPE"});
        //device_infos.insert({CL_DEVICE_VENDOR, InfoTypeInfo("CL_DEVICE_VENDOR"});
        //device_infos.insert({CL_DEVICE_VENDOR_ID, InfoTypeInfo("CL_DEVICE_VENDOR_ID"});
        //device_infos.insert({CL_DEVICE_VERSION, InfoTypeInfo("CL_DEVICE_VERSION"});
        //device_infos.insert({CL_DRIVER_VERSION, InfoTypeInfo("CL_DRIVER_VERSION"});
        clGetDeviceIDs(*(platform_id_ptr + p), CL_DEVICE_TYPE_ALL, NULL, nullptr, &dnum);
        cl_device_id* device_id_ptr = (cl_device_id*)malloc(dnum * sizeof(cl_device_id));
        clGetDeviceIDs(*(platform_id_ptr + p), CL_DEVICE_TYPE_ALL, dnum, device_id_ptr, NULL);
        for(int d = 0; d < dnum; ++d){
            info += std::string("   |\n    ->") + "Device " + std::to_string(d) + "\n";
            device_infos.reserve(device_infos.size());
            for(auto info_item = device_infos.begin(); info_item != device_infos.end(); ++info_item){
                info += get_device_info(info_item->first, info_item->second, *(device_id_ptr + d), &device_info_size, &device_info);
            }
        }
    }

    delete platform_id_ptr;
    std::cout << info << std::endl;
    return 0;
}

int main(int argc, int* argv[]){
    return get_info();
};