#ifdef __cplusplus
#include <memory>
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
    CLDEVICEEXECCAPABILITIES,
    CLULONG,
    CLDEVICEMEMCACHETYPE,
    SIZET,
    CLDEVICELOCALMEMTYPE,
    BUFFER,
    POINTER,
    CLDEVICEPARTITIONPROPERTIES,
    CLDEVICEAFFINITYDOMAIN,
    CLCOMMANDQUEUEPROPERTIES,
    CLDEVICETYPE,
    TheFlag,
};

class InfoTypeInfo{
    public:
        std::string _name;
        InfoType _type;
        std::string _help;
        std::vector<std::string> _bit_field_name;
    public:
        typedef std::vector<std::string> BitFieldNameType;
    public:
        InfoTypeInfo(
            const std::string& name,
            InfoType type,
            const std::string& help="",
            const BitFieldNameType& bit_field_name=BitFieldNameType(),
            InfoType flag=TheFlag)
            : _name(name), 
            _type(type),
            _help(help), 
            _bit_field_name(bit_field_name)
        {};
        virtual ~InfoTypeInfo(){};
    public:
        virtual void* get_options_name() {
            return nullptr;
        };
    public:
        virtual InfoType get_buffer_type(){
            return InfoType();
        };
        virtual size_t get_buffer_size(){
            return NULL;
        };
        virtual void set_buffer_size(size_t size){
            return;
        }
        virtual std::string get_pointer_name(){
            return "";
        }
};

class _InfoTypeInfoPointer : public InfoTypeInfo{
    public:
        std::string _pointer_name;
    public:
        _InfoTypeInfoPointer(
            const std::string& name,
            InfoType type,
            const std::string& help="",
            const BitFieldNameType& bit_field_name=BitFieldNameType(),
            const std::string& pointer_name="None",
            InfoType flag=TheFlag)
            : InfoTypeInfo(name, type, help, bit_field_name, TheFlag),
            _pointer_name(pointer_name)
        {};
    public:
        virtual std::string get_pointer_name(){
            return _pointer_name;
        }
};

/**
 * \brief brief
 * \note note 用于Buffer类型，也就是说：需要知道单元类型与buffer长度，如size_t[],cl_uint[]等信息，
 * 但是char[]得看文档，看看是表示char*，还是buffer(char*, size)
 * \param[in] size size_t buffer的单元个数
 * \param[in] buffer_type InfoType buffer单元数据类型
 * \author none
 * \since version
 * */
class _InfoTypeInfoBuffer : public InfoTypeInfo{
    public:
        size_t _size;
        InfoType _buffer_type;
    public:
        _InfoTypeInfoBuffer(
            const std::string& name,
            InfoType type,
            const std::string& help="",
            const BitFieldNameType& bit_field_name=BitFieldNameType(),
            size_t size=1,
            InfoType buffer_type=CLUINT,
            InfoType flag=TheFlag
        )
            : InfoTypeInfo(name, type, help, bit_field_name, flag),
            _size(size),
            _buffer_type(buffer_type)
            {};
        virtual ~_InfoTypeInfoBuffer(){};
    public:
        virtual InfoType get_buffer_type(){
            return _buffer_type;
        };
        virtual size_t get_buffer_size(){
            return _size;
        };
        virtual void set_buffer_size(size_t size){
            _size = size;
        }
}; 

/**
 * \brief brief
 * \note note 用于表示需要有单元名称与数据的类型，比如某些信息是几个类别只有一个成立，这时候就需要传入{信息类型，信息名称}，相当于匹配类型信息
 * \param[in] options_name map {OptionType, std::string} string 用来输出字符，OptionType用来
 * \author none
 * \since version
 * */
template<typename OptionType> class _InfoTypeInfo : public InfoTypeInfo{
    public:
        typedef std::map<OptionType, std::string> OptionsNameType;
    public:
        OptionsNameType _options_name;
    public:
        _InfoTypeInfo<OptionType>(
            const std::string& name,
            InfoType type,
            const std::string& help="",
            const BitFieldNameType& bit_field_name=BitFieldNameType(),
            const OptionsNameType& options_name=OptionsNameType(),
            InfoType flag=TheFlag)
            : InfoTypeInfo(name, type, help, bit_field_name, flag),
            _options_name(options_name)
        {};
        virtual ~_InfoTypeInfo<OptionType>(){};
    public:
        virtual void* get_options_name() {
            return &_options_name;
        };
};

#define Default_Info_Format(INFO_TYPE_INFO, TYPE) \
std::string("       ") + INFO_TYPE_INFO->_name + "("  + TYPE + ")" + "[" + INFO_TYPE_INFO->_help + "]" + ":\n" + "         "

#define ToStringDeviceInfoFormat(INFO_TYPE_INFO, TYPE, DEVICE_INFO_PPTR) \
Default_Info_Format(INFO_TYPE_INFO, #TYPE) + std::to_string(*(TYPE*)*DEVICE_INFO_PPTR) + "\n";

#define PtrStringDeviceInfoFormat(INFO_TYPE_INFO, TYPE, DEVICE_INFO_PPTR) \
Default_Info_Format(INFO_TYPE_INFO, #TYPE) + std::string((TYPE*)*DEVICE_INFO_PPTR) + "\n";

#define BitDeviceInfoFormat(INFO_TYPE_INFO, TYPE, DEVICE_INFO_PPTR) \
std::string bit_field_name; \
std::vector<int> char_num; \
for(auto bit_field_name_item = INFO_TYPE_INFO->_bit_field_name.begin(); bit_field_name_item != INFO_TYPE_INFO->_bit_field_name.end(); ++bit_field_name_item){ \
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

#define OptionsDeviceInfoFormat(INFO_TYPE_INFO, TYPE, DEVICE_INFO_PPTR) \
Default_Info_Format(INFO_TYPE_INFO, #TYPE) + ((_InfoTypeInfo<TYPE>::OptionsNameType*)INFO_TYPE_INFO->get_options_name())->find(*(TYPE*)*DEVICE_INFO_PPTR)->second + "\n";

#define BufferDeviceInfoFormat(INFO_TYPE_INFO, TYPE, DEVICE_INFO_PPTR, SIZE) \
std::string temp = "["; \
for(size_t i = 0; i < SIZE; ++i){ \
    temp += std::to_string(*(*(TYPE**)(DEVICE_INFO_PPTR) + i)) + (i != (SIZE - 1) ? ", " : "]"); \
} \
temp = Default_Info_Format(INFO_TYPE_INFO, #TYPE) + temp + "\n";

#define PointerDeviceInfoFormat(INFO_TYPE_INFO, TYPE, DEVICE_INFO_PPTR) \
Default_Info_Format(INFO_TYPE_INFO, TYPE) + std::to_string(reinterpret_cast<std::uintptr_t>(*DEVICE_INFO_PPTR)) + "\n";

std::string get_device_info(cl_device_info info, std::shared_ptr<InfoTypeInfo> info_type_info, \
cl_device_id device_id, size_t* device_info_size, cl_device_info** device_info_pptr){
    clGetDeviceInfo(device_id, info, NULL, nullptr, device_info_size);
    *device_info_pptr = (cl_device_info*)malloc(*device_info_size);
    clGetDeviceInfo(device_id, info, *device_info_size, *device_info_pptr, NULL);
    switch(info_type_info->_type){
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
        case CLDEVICEEXECCAPABILITIES:{
            BitDeviceInfoFormat(info_type_info, cl_device_fp_config, device_info_pptr);
            return temp;
            break;
        }
        case CLULONG:{
            return ToStringDeviceInfoFormat(info_type_info, cl_bool, device_info_pptr);
            break;
        }
        case CLDEVICEMEMCACHETYPE:{
            return OptionsDeviceInfoFormat(info_type_info, cl_device_mem_cache_type, device_info_pptr);
            break;
        }
        case SIZET:{
            return ToStringDeviceInfoFormat(info_type_info, size_t, device_info_pptr);
            break;
        }
        case CLDEVICELOCALMEMTYPE:{
            return OptionsDeviceInfoFormat(info_type_info, cl_device_local_mem_type, device_info_pptr);
            break;
        }
        case BUFFER:{
            switch(info_type_info->get_buffer_type()){
                case SIZET:{
                    BufferDeviceInfoFormat(info_type_info, size_t, device_info_pptr, info_type_info->get_buffer_size());
                    return temp;
                    break;
                }
                default:{
                    break;
                }
            }
            //BufferDeviceInfoFormat(info_type_info, info_type_info->)
            break;
        }
        case POINTER:{
            return PointerDeviceInfoFormat(info_type_info, info_type_info->get_pointer_name(), device_info_pptr);
            break;
        }
        case CLDEVICEPARTITIONPROPERTIES:{
            break;
        }
        case CLDEVICEAFFINITYDOMAIN:{
            BitDeviceInfoFormat(info_type_info, cl_device_fp_config, device_info_pptr);
            return temp;
            break;
        }
        case CLCOMMANDQUEUEPROPERTIES:{
            BitDeviceInfoFormat(info_type_info, cl_command_queue_properties, device_info_pptr);
            return temp;
            break;
        }
        case CLDEVICETYPE:{
            return OptionsDeviceInfoFormat(info_type_info, cl_device_type, device_info_pptr);
            break;
        }
        default:{
            return std::string("       ") + info_type_info->_name + "(**)" + ":\n" + "         type not defined\n";
            break;
        }
    }
    delete *device_info_pptr;
    return "";
};

cl_uint get_dimensions(cl_device_id device_id, int type){
    size_t size;
    cl_device_info* device_info_ptr;
    clGetDeviceInfo(device_id, type, NULL, nullptr, &size);
    device_info_ptr = (cl_device_info*)malloc(size);
    clGetDeviceInfo(device_id, type, size, device_info_ptr, NULL);
    cl_uint temp = *(cl_uint*)device_info_ptr;
    delete (cl_uint*)device_info_ptr;
    return temp;
};

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

    cl_uint dnum = NULL;
    //cl_device_id* device_id_ptr = nullptr;

    size_t platform_info_size = 0;
    cl_platform_info* platform_info = nullptr;
    size_t device_info_size = 0;
    cl_device_info* device_info = nullptr;
    std::string info = "";
    for(unsigned int p = 0; p < pnum; ++p){
        info += "Platform " + std::to_string(p) + "\n";
        for(auto platform_info_item = platform_info_names.begin(); platform_info_item != platform_info_names.end(); ++platform_info_item){
            clGetPlatformInfo(*(platform_id_ptr + p), platform_info_item->first, NULL, nullptr, &platform_info_size);
            platform_info = (cl_platform_info*)malloc(platform_info_size);
            clGetPlatformInfo(*(platform_id_ptr + p), platform_info_item->first, platform_info_size, platform_info, NULL);
            info += "   " + platform_info_item->second + ":\n";
            info += "       " + std::string((char*)platform_info) + "\n";
            delete platform_info;
        }
        //std::map<int, InfoTypeInfo> device_infos;
        std::vector<int> device_info_type;
        std::vector<std::shared_ptr<InfoTypeInfo>> device_infos;
        device_info_type.push_back(CL_DEVICE_ADDRESS_BITS);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_ADDRESS_BITS", CLUINT)));
        device_info_type.push_back(CL_DEVICE_AVAILABLE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_AVAILABLE", CLBOOL)));
        device_info_type.push_back(CL_DEVICE_BUILT_IN_KERNELS);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_BUILT_IN_KERNELS", CHAR, "a set of supported built-in kernels")));
        device_info_type.push_back(CL_DEVICE_COMPILER_AVAILABLE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_COMPILER_AVAILABLE", CLBOOL)));
        device_info_type.push_back(CL_DEVICE_DOUBLE_FP_CONFIG);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_DOUBLE_FP_CONFIG", CLDEVICEFPCONFIG, "", {"CL_FP_FMA", "CL_FP_ROUND_TO_NEAREST" ,"CL_FP_ROUND_TO_ZERO", "CL_FP_ROUND_TO_INF", "CL_FP_INF_NAN" ,"CL_FP_DENORM"})));
        device_info_type.push_back(CL_DEVICE_ENDIAN_LITTLE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_ENDIAN_LITTLE", CLBOOL, "Is CL_TRUE if the OpenCL device is a little endian device and CL_FALSE otherwise.")));
        device_info_type.push_back(CL_DEVICE_ERROR_CORRECTION_SUPPORT);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_ERROR_CORRECTION_SUPPORT", CLBOOL)));
        device_info_type.push_back(CL_DEVICE_EXECUTION_CAPABILITIES);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_EXECUTION_CAPABILITIES", CLDEVICEEXECCAPABILITIES, "", {"CL_EXEC_KERNEL", "CL_EXEC_NATIVE_KERNEL"})));
        device_info_type.push_back(CL_DEVICE_EXTENSIONS);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_EXTENSIONS", CHAR)));
        device_info_type.push_back(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_GLOBAL_MEM_CACHE_SIZE", CLULONG)));
        device_info_type.push_back(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE);
        device_infos.push_back(std::shared_ptr<_InfoTypeInfo<cl_device_mem_cache_type>>(new _InfoTypeInfo<cl_device_mem_cache_type>("CL_DEVICE_GLOBAL_MEM_CACHE_TYPE", CLDEVICEMEMCACHETYPE, "", _InfoTypeInfo<cl_device_mem_cache_type>::BitFieldNameType(), _InfoTypeInfo<cl_device_mem_cache_type>::OptionsNameType({{CL_NONE, "CL_NONE"}, {CL_READ_ONLY_CACHE, "CL_READ_ONLY_CACHE"}, {CL_READ_WRITE_CACHE, "CL_READ_WRITE_CACHE"}}))));
        device_info_type.push_back(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE", CLUINT, "bytes")));
        device_info_type.push_back(CL_DEVICE_GLOBAL_MEM_SIZE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_GLOBAL_MEM_SIZE", CLULONG, "bytes")));
        //device_insme.push_back(CL_DEVICE_HALF_FP_CONFIG, std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_HALF_FP_CONFIG",)));
        device_info_type.push_back(CL_DEVICE_HOST_UNIFIED_MEMORY);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_HOST_UNIFIED_MEMORY", CLBOOL, "Is CL_TRUE if the device and the host have a unified memory subsystem")));
        device_info_type.push_back(CL_DEVICE_IMAGE_SUPPORT);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_IMAGE_SUPPORT", CLBOOL)));
        device_info_type.push_back(CL_DEVICE_IMAGE2D_MAX_HEIGHT);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_IMAGE2D_MAX_HEIGHT", SIZET)));
        device_info_type.push_back(CL_DEVICE_IMAGE2D_MAX_WIDTH);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_IMAGE2D_MAX_WIDTH", SIZET)));
        device_info_type.push_back(CL_DEVICE_IMAGE3D_MAX_DEPTH);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_IMAGE3D_MAX_DEPTH", SIZET)));
        device_info_type.push_back(CL_DEVICE_IMAGE3D_MAX_HEIGHT);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_IMAGE3D_MAX_HEIGHT", SIZET)));
        device_info_type.push_back(CL_DEVICE_IMAGE3D_MAX_WIDTH);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_IMAGE3D_MAX_WIDTH", SIZET)));
        device_info_type.push_back(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_IMAGE_MAX_BUFFER_SIZE", SIZET)));
        device_info_type.push_back(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_IMAGE_MAX_ARRAY_SIZE", SIZET)));
        device_info_type.push_back(CL_DEVICE_LINKER_AVAILABLE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_LINKER_AVAILABLE", CLBOOL)));
        device_info_type.push_back(CL_DEVICE_LOCAL_MEM_SIZE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_LOCAL_MEM_SIZE", CLULONG)));
        device_info_type.push_back(CL_DEVICE_LOCAL_MEM_TYPE);
        device_infos.push_back(std::shared_ptr<_InfoTypeInfo<cl_device_local_mem_type>>(new _InfoTypeInfo<cl_device_local_mem_type>("CL_DEVICE_LOCAL_MEM_TYPE", CLDEVICELOCALMEMTYPE, "", InfoTypeInfo::BitFieldNameType(), _InfoTypeInfo<cl_device_local_mem_type>::OptionsNameType({{CL_LOCAL, "CL_LOCAL"}, {CL_GLOBAL, "CL_GLOBAL"}, {CL_NONE, "CL_NONE"}}))));
        device_info_type.push_back(CL_DEVICE_MAX_CLOCK_FREQUENCY);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MAX_CLOCK_FREQUENCY", CLUINT)));
        device_info_type.push_back(CL_DEVICE_MAX_COMPUTE_UNITS);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MAX_COMPUTE_UNITS", CLUINT)));
        device_info_type.push_back(CL_DEVICE_MAX_CONSTANT_ARGS);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MAX_CONSTANT_ARGS", CLUINT, "Max number of arguments declared with the __constant qualifier in a kernel. The minimum value is 8 for devices that are not of type CL_DEVICE_TYPE_CUSTOM.")));
        device_info_type.push_back(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE", CLULONG)));
        device_info_type.push_back(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MAX_MEM_ALLOC_SIZE", CLULONG, "Max size of memory object allocation in bytes. The minimum value is max (1/4th of CL_DEVICE_GLOBAL_MEM_SIZE, 128*1024*1024) for devices that are not of type CL_DEVICE_TYPE_CUSTOM.")));
        device_info_type.push_back(CL_DEVICE_MAX_PARAMETER_SIZE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MAX_PARAMETER_SIZE", SIZET, "Max size in bytes of the arguments that can be passed to a kernel. The minimum value is 1024 for devices that are not of type CL_DEVICE_TYPE_CUSTOM. For this minimum value, only a maximum of 128 arguments can be passed to a kernel.")));
        device_info_type.push_back(CL_DEVICE_MAX_READ_IMAGE_ARGS);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MAX_READ_IMAGE_ARGS", CLUINT, "Max number of simultaneous image objects that can be read by a kernel. The minimum value is 128 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.")));
        device_info_type.push_back(CL_DEVICE_MAX_SAMPLERS);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MAX_SAMPLERS", CLUINT, "Maximum number of samplers that can be used in a kernel. The minimum value is 16 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE. (Also see sampler_t.)")));
        device_info_type.push_back(CL_DEVICE_MAX_WORK_GROUP_SIZE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MAX_WORK_GROUP_SIZE", SIZET, "Maximum number of work-items in a work-group executing a kernel on a single compute unit, using the data parallel execution model. (Refer to clEnqueueNDRangeKernel). The minimum value is 1.")));
        device_info_type.push_back(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS", CLUINT, "Maximum dimensions that specify the global and local work-item IDs used by the data parallel execution model. (Refer to clEnqueueNDRangeKernel). The minimum value is 3 for devices that are not of type CL_DEVICE_TYPE_CUSTOM.")));
        device_info_type.push_back(CL_DEVICE_MAX_WORK_ITEM_SIZES);
        device_infos.push_back(std::shared_ptr<_InfoTypeInfoBuffer>(new _InfoTypeInfoBuffer("CL_DEVICE_MAX_WORK_ITEM_SIZES", BUFFER, "", InfoTypeInfo::BitFieldNameType(), NULL, SIZET)));
        device_info_type.push_back(CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MAX_WRITE_IMAGE_ARGS", CLUINT, "Max number of simultaneous image objects that can be written to by a kernel. The minimum value is 8 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.")));
        device_info_type.push_back(CL_DEVICE_MEM_BASE_ADDR_ALIGN);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MEM_BASE_ADDR_ALIGN", CLUINT, "The minimum value is the size (in bits) of the largest OpenCL built-in data type supported by the device (long16 in FULL profile, long16 or int16 in EMBEDDED profile) for devices that are not of type CL_DEVICE_TYPE_CUSTOM.")));
        device_info_type.push_back(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE", CLUINT, "Deprecated in OpenCL 1.2. The smallest alignment in bytes which can be used for any data type.")));
        device_info_type.push_back(CL_DEVICE_NAME);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_NAME", CHAR, "CL_DEVICE_NAME")));
        device_info_type.push_back(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR", CLUINT)));
        device_info_type.push_back(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT", CLUINT)));
        device_info_type.push_back(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_INT", CLUINT)));
        device_info_type.push_back(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG", CLUINT)));
        device_info_type.push_back(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT", CLUINT)));
        device_info_type.push_back(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE", CLUINT)));
        device_info_type.push_back(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF", CLUINT)));
        device_info_type.push_back(CL_DEVICE_OPENCL_C_VERSION);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_OPENCL_C_VERSION", CHAR)));
        device_info_type.push_back(CL_DEVICE_PARENT_DEVICE);
        device_infos.push_back(std::shared_ptr<_InfoTypeInfoPointer>(new _InfoTypeInfoPointer("CL_DEVICE_PARENT_DEVICE", POINTER, "", InfoTypeInfo::BitFieldNameType(), "cl_device_parent_device")));
        device_info_type.push_back(CL_DEVICE_PARTITION_MAX_SUB_DEVICES);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PARTITION_MAX_SUB_DEVICES", CLUINT)));
        //device_info_type.push_back(CL_DEVICE_PARTITION_PROPERTIES);
        //device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PARTITION_PROPERTIES", CLDEVICEPARTITIONPROPERTIES)));
        device_info_type.push_back(CL_DEVICE_PARTITION_AFFINITY_DOMAIN);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PARTITION_AFFINITY_DOMAIN", CLDEVICEAFFINITYDOMAIN, "", {"CL_DEVICE_AFFINITY_DOMAIN_NUMA", "CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE", "CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE", "CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE", "CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE", "CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE"})));
        //device_info_type.push_back(CL_DEVICE_PARTITION_TYPE);
        //device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PARTITION_TYPE",)));
        device_info_type.push_back(CL_DEVICE_PLATFORM);
        device_infos.push_back(std::shared_ptr<_InfoTypeInfoPointer>(new _InfoTypeInfoPointer("CL_DEVICE_PLATFORM", POINTER, "", InfoTypeInfo::BitFieldNameType(), "cl_device_platform")));
        device_info_type.push_back(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR", CLUINT)));
        device_info_type.push_back(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT", CLUINT)));
        device_info_type.push_back(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT", CLUINT)));
        device_info_type.push_back(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG", CLUINT)));
        device_info_type.push_back(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT", CLUINT)));
        device_info_type.push_back(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE", CLUINT)));
        device_info_type.push_back(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF", CLUINT)));
        device_info_type.push_back(CL_DEVICE_PRINTF_BUFFER_SIZE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PRINTF_BUFFER_SIZE", SIZET)));
        device_info_type.push_back(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PREFERRED_INTEROP_USER_SYNC", CLBOOL, "Is CL_TRUE if the device's preference is for the user to be responsible for synchronization, when sharing memory objects between OpenCL and other APIs such as DirectX, CL_FALSE if the device / implementation has a performant path for performing synchronization of memory object shared between OpenCL and other APIs such as DirectX")));
        device_info_type.push_back(CL_DEVICE_PROFILE);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PROFILE", CHAR)));
        device_info_type.push_back(CL_DEVICE_PROFILING_TIMER_RESOLUTION);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_PROFILING_TIMER_RESOLUTION", SIZET)));
        device_info_type.push_back(CL_DEVICE_QUEUE_PROPERTIES);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_QUEUE_PROPERTIES", CLCOMMANDQUEUEPROPERTIES, 
        "Describes the command-queue properties supported by the device. This is a bit-field that describes one or more of the following values:CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE-CL_QUEUE_PROFILING_ENABLE These properties are described in the table for clCreateCommandQueue. The mandated minimum capability is CL_QUEUE_PROFILING_ENABLE.", 
        InfoTypeInfo::BitFieldNameType({"CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE", "CL_QUEUE_PROFILING_ENABLE"}))));
        device_info_type.push_back(CL_DEVICE_REFERENCE_COUNT);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_REFERENCE_COUNT", CLUINT, "Returns the device reference count. If the device is a root-level device, a reference count of one is returned.")));
        device_info_type.push_back(CL_DEVICE_SINGLE_FP_CONFIG);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_SINGLE_FP_CONFIG", CLDEVICEFPCONFIG, "", {"CL_FP_DENORM", "CL_FP_INF_NAN", "CL_FP_ROUND_TO_NEAREST", "CL_FP_ROUND_TO_ZERO", "CL_FP_ROUND_TO_INF", "CL_FP_FMA", "CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT", "CL_FP_SOFT_FLOAT"})));
        device_info_type.push_back(CL_DEVICE_TYPE);
        device_infos.push_back(std::shared_ptr<_InfoTypeInfo<cl_device_type>>(new _InfoTypeInfo<cl_device_type>("CL_DEVIC, E_TYPE", CLDEVICETYPE, "", InfoTypeInfo::BitFieldNameType(), {{CL_DEVICE_TYPE_CPU, "CL_DEVICE_TYPE_CPU"}, {CL_DEVICE_TYPE_GPU, "CL_DEVICE_TYPE_GPU"}, {CL_DEVICE_TYPE_ACCELERATOR, "CL_DEVICE_TYPE_ACCELERATOR"}, {CL_DEVICE_TYPE_ACCELERATOR, "CL_DEVICE_TYPE_ACCELERATOR"}})));
        device_info_type.push_back(CL_DEVICE_VENDOR);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_VENDOR", CHAR)));
        device_info_type.push_back(CL_DEVICE_VENDOR_ID);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_VENDOR_ID", CLUINT)));
        device_info_type.push_back(CL_DEVICE_VERSION);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DEVICE_VERSION", CHAR)));
        device_info_type.push_back(CL_DRIVER_VERSION);
        device_infos.push_back(std::shared_ptr<InfoTypeInfo>(new InfoTypeInfo("CL_DRIVER_VERSION", CHAR)));
        clGetDeviceIDs(*(platform_id_ptr + p), CL_DEVICE_TYPE_ALL, NULL, nullptr, &dnum);
        cl_device_id* device_id_ptr = (cl_device_id*)malloc(dnum * sizeof(cl_device_id));
        clGetDeviceIDs(*(platform_id_ptr + p), CL_DEVICE_TYPE_ALL, dnum, device_id_ptr, NULL);
        for(unsigned int d = 0; d < dnum; ++d){
            info += std::string("   |\n    ->") + "Device " + std::to_string(d) + "(" + std::to_string(reinterpret_cast<std::uintptr_t>(platform_id_ptr + d)) + ")" + "\n";
            device_infos.reserve(device_infos.size());
            for(size_t i = 0; i < device_info_type.size(); ++i){
                switch(device_infos[i]->_type){
                    case BUFFER:{
                        switch(device_info_type[i]){
                            case CL_DEVICE_MAX_WORK_ITEM_SIZES:{
                                device_infos[i]->set_buffer_size(get_dimensions(*(device_id_ptr + d), CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));
                                break;
                            };
                            default:{
                                break;
                            };
                        }
                        break;
                    }
                    default:{
                        break;
                    }
                }
                info += get_device_info(device_info_type[i], device_infos[i], *(device_id_ptr + d), &device_info_size, &device_info);
            }
        }
    }

    delete platform_id_ptr;
    std::cout << info << std::endl;
    return 0;
}

int main(int argc, char* argv[]){
    return get_info();
};