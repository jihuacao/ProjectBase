# base on glog v2.2.2
include(popular_message)
include(external_setting)
cmakelists_base_header()
project_base_system_message()
include(fold_operation)

set(module protobuf)
set(${module}_target_name protobuf)
set(${module}_supported_version 3.5.0)
version_selector(${module} ${module}_supported_version 3.5.0)

message(STATUS "tt:${external_install_path}")
find_package(${module} ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${external_install_path})

if(${${module}_FOUND})
    message(STATUS "${module} Found for  ${${module}_version}")
else()
    include(ExternalProject)
endif()
