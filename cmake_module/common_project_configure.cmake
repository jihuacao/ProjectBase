#[[
工程上比较固定通用的设置
在最顶层调用一次
]]
function(project_common_setting)
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