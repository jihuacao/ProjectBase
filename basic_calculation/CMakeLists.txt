## compile basic_calculation 
include(popular_message)
cmakelists_base_header()

include(obj_name_provide)
set(module BasicCalculation)
set(basic_calculation_module_name ${module} CACHE INTERNAL "module name")
obj_name_provide(CMAKE_BUILD_TYPE project_base_build_shared_lib module)

file(GLOB ${module}_cpp_src *.h *.cc *.hpp *.cpp)
if(cuda_FOUND)
    file(GLOB ${module}_cu_src *.cu)
endif()
if(OpenCL_FOUND)
    file(GLOB ${module}_cl_src *.cmc *.cl)
endif()

set(CMAKE_CXX_FLAGS "-DPROJECT_BASE_BASIC_CALCULATION=1 ${CMAKE_CXX_FLAGS}")

#生成链接库
#add_library(${${module}_target_name} ${${module}_build_link_type} ${${module}_cpp_src} ${${module}_cl_src} ${${module}_cu_src})
set(SRC
    "./basic_calculation.cpp"
    "./basic_calculation.hpp"
    "./col2im.hpp"
    "./col2im.cpp"
    "./Define.hpp"
    #"./col2im_kernel.cl"
    #"./col2im.cmc"
)
add_library(${${module}_target_name} ${${module}_build_link_type} ${SRC})
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

