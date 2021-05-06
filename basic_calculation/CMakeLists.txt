## compile basic_calculation 
include(popular_message)
cmakelists_base_header()

include(obj_name_provide)
set(module BasicCalculation)
set(basic_calculation_module_name ${module} CACHE INTERNAL "module name")
obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib module)

#生成链接库
if (OPENCL)
    find_package(OpenCL REQUIRED)
    if (OpenCL_FOUND)
        message(STATUS "found opencl")
        set(${${cm_module_name}_target_name} opencl)
        add_library(${${cm_module_name}_target_name} INTERFACE IMPORTED)
        set_property(TARGET ${${cm_module_name}_target_name} PROPERTY
        	INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_INCLUDE_DIRS})
        set_property(TARGET ${${cm_module_name}_target_name} PROPERTY
            INTERFACE_LINK_LIBRARIES ${OpenCL_LIBRARY})
        list (APPEND DARKNET_DEFINITIONS GPU CL_TARGET_OPENCL_VERSION=120)
        set(CMAKE_CXX_FLAGS "-DCOMPUTE_MACHINE -DOPENCL")
    else()
        message(STATUS "specify with opencl but non opencl found")
        add_library(${${cm_module_name}_target_name} INTERFACE IMPORTED)
    endif()
else()
if (CUDA)
else()

file(GLOB test_${module}_cpp_src *.h *.cc *.hpp *.cpp)
file(GLOB test_${module}_cl_src *.cmc *.cl)
file(GLOB test_${module}_cu_src *.cmc *.cu) 

set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_BASIC_CALCULATION=1 ${CMAKE_CXX_FLAGS}")

add_library(${${module}_target_name} ${${module}_build_link_type} ${DIR_LIB_SRCS})
target_link_libraries(${${module}_target_name} ${${cm_module_name}_target_name})

## install basic_calculation 
message(STATUS ${CMAKE_CURRENT_LIST_FILE})
set(module basic_calculation)
file(
    GLOB
    ${module}_files
    *.hpp
)
install(FILES ${${module}_head} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${module})

include(test_engineer_config)
build_test_option(${module})
if(${_build_${module}_test})
    add_subdirectory(basic_calculation_test)
else()
endif()

