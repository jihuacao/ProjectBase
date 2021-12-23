#[[
    工程上比较固定通用的设置
    在最顶层调用一次
    * 确定后缀
        * library: ${${lib_postfix_var_name}}
    * 确定公共编译参数
]]
function(project_common_setting)
    #[[
        不同操作系统的设置
    ]]
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
    else()
    endif()
    #[[
        不同构建工具与不同平台的设置
            * x86-64
            * x86-32
            * arm
            * apple
    ]]
    if(${CMAKE_GENERATOR} MATCHES "Visual Studio")
        #[[
            * 设置编译参数
                * 需要指定/MD /MDd /MT /MTd /ML /MLd以及对应的object后缀
            * 设置后缀
        ]]
        set(MSVC_RT_TYPE "MD" CACHE STRING "specifical build the gtest in shared or follow in external_build_shared")
        set_property(CACHE MSVC_RT_TYPE PROPERTY STRINGS "MD" "MT" "ML")
        if(${MSVC_RT_TYPE} STREQUAL "MD")
            set(_lib_postfix "${_lib_postfix}md")
        elseif(${MSVC_RT_TYPE} STREQUAL "MT")
            set(_lib_postfix "${_lib_postfix}mt")
        elseif(${MSVC_RT_TYPE} STREQUAL "ml")
            set(_lib_postfix "${_lib_postfix}ml")
        else()
        endif()
        #[[
            设置前缀
        ]]
        if(${CMAKE_GENERATOR_PLATFORM} MATCHES "x64")
            set(CMAKE_SHARED_LIBRARY_PREFIX "")
            set(CMAKE_IMPORT_LIBRARY_PREFIX "")
            set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
        elseif(${CMAKE_GENERATOR_PLATFORM} MATCHES "ARM")
        else()
        endif()
    elseif(${CMAKE_GENERATOR} MATCHES "Makefiles")
    else()
    endif()
    #[[
    设置四个配置类型
    * DEBUG
    * RELEASE
    * RELWITHDEBINFO
    * MINSIZEREL
    四个类型文件
    * PDB for windows
    * RUNTIME
    * ARCHIVE
    * RUNTIME
    的输出路径
    ]]
    foreach(configure "DEBUG" "RELEASE" "RELWITHDEBINFO" "MINSIZEREL")
        string(TOLOWER ${configure} lower_configure)
        message(STATUS "configure ${configure} to ${lower_configure}")
        if(SYSTEM_NAME MATCHES "windows")
            set(CMAKE_PDB_OUTPUT_DIRECTORY_${configure} ${CMAKE_BINARY_DIR}/${lower_configure}/lib PARENT_SCOPE)
        endif()
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${configure} ${CMAKE_BINARY_DIR}/${lower_configure}/bin PARENT_SCOPE)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${configure} ${CMAKE_BINARY_DIR}/${lower_configure}/lib PARENT_SCOPE)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${configure} ${CMAKE_BINARY_DIR}/${lower_configure}/bin PARENT_SCOPE)
    endforeach()
endfunction(project_common_setting)

function(common_target_config target)
    #set_target_properties(
    #    ${target}
    #    PROPERTIES
    #)
endfunction(common_target_config)