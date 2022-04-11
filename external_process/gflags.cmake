# base on gflags v2.2.2
function(gflags_target)
    include(popular_message)
    include(ExternalProject)
    include(external_setting)
    include(fold_operation)
    set(module gflags)

    set(gflags_url https://github.com/gflags/gflags.git)
    set(${module}_supported_version "2.2.2")
    set(${module}_supported_tag "v2.2.2")
    version_selector(${module} ${module}_supported_version 2.2.2)
    version_tag_matcher(${module} ${module}_supported_version ${module}_supported_tag ${module}_version)
    #default_external_project_build_type(${module})
    #project_build_shared(${module})
    cmake_external_project_common_args(${module})

    message("gflags: version->${${module}_version} build in->${_${module}_build_type} shared?->${_${module}_build_shared}")

    set(GFLAGS_USE_TARGET_NAMESPACE TRUE)
    option(${module}_need_nothreads "need gflags nothreads or not" OFF)
    set(GFLAGS_NOTHREADS ${gflags_need_nothreads})
    find_package(${module} ${${module}_version} CONFIG NO_CMAKE_PACKAGE_REGISTRY PATHS ${${module}_cmake_install_prefix})

    if(${module}_FOUND)
        message(STATUS "FOUND GFLAGS_INCLUDE_DIR: ${GFLAGS_INCLUDE_DIR}")
        add_library(interface_lib${module} INTERFACE)
        target_link_libraries(interface_lib${module} INTERFACE gflags::gflags)
    else()
        ExternalProject_Add(
            ext_${module}
            "${${module}_external_project_add_args}"
            PREFIX ${module} 
            "${${module}_external_project_add_git_args}"
            GIT_REPOSITORY ${${module}_url}
            GIT_TAG ${${module}_tag}
            CMAKE_ARGS
                "${${module}_cmake_args}"
            CMAKE_CACHE_ARGS
                -DBUILD_SHARED_LIBS:BOOL=${${${module}_build_shared_var_name}}
                -DBUILD_PACKAGING:BOOL=ON
                -DBUILD_TESTING:BOOL=OFF
                -DBUILD_gflags_LIB:BOOL=ON
                -DBUILD_gflags_nothreads_LIB:BOOL=${${module}_need_nothreads}
                -DREGISTER_BUILD_DIR:BOOL=OFF
                -DREGISTER_INSTALL_PREFIX:BOOL=OFF
        )
        set(${module}_include_dir ${${module}_cmake_install_prefix}/include)
        set(${module}_lib_dir ${${module}_cmake_install_prefix}/lib)
        touch_fold(${module}_include_dir)
        touch_fold(${module}_lib_dir)

        list(
            APPEND
            status_registry
            SYSTEM_NAME 
            GENERATOR 
            GENERATOR_PLATFORM
            GENERATOR_TOOLSET
            GENERATOR_INSTANCE
            BUILD_SHARED
            BUILD_TYPE
        )
        list(
            APPEND
            component_registry
            PREFIX
            POSTFIX
            EXTENSION
            DEFINITIONS
            LIBS
        )
        generate_object_name_component(
            status_registry
            component_registry
            SYSTEM_NAME ${CMAKE_SYSTEM_NAME}
            GENERATOR ${${module}_cmake_generator}
            GENERATOR_PLATFORM ${${module}_cmake_generator_platform}
            GENERATOR_TOOLSET ${${module}_cmake_generator_toolset}
            GENERATOR_INSTANCE ${${module}_cmake_generator_instance}
            BUILD_SHARED ${${${module}_build_shared_var_name}}
            BUILD_TYPE ${${${module}_build_type_var_name}}
            SYSTEM_NAME_LIST 
            Windows 
            Windows 
            Windows 
            Windows
            Linux
            Linux
            Linux
            Linux
            CYGWIN
            CYGWIN
            CYGWIN
            CYGWIN
            GENERATOR_LIST 
            "Visual Studio" 
            "Visual Studio" 
            "Visual Studio"
            "Visual Studio"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            "Unix Makefiles"
            GENERATOR_PLATFORM_LIST x64 x64 x64 x64 ANY ANY ANY ANY ANY ANY ANY ANY
            GENERATOR_TOOLSET_LIST ANY ANY ANY ANY ANY ANY ANY ANY ANY ANY ANY ANY
            GENERATOR_INSTANCE_LIST ANY ANY ANY ANY ANY ANY ANY ANY ANY ANY ANY ANY
            BUILD_SHARED_LIST ON ON OFF OFF ON ON OFF OFF ON ON OFF OFF
            BUILD_TYPE_LIST RELEASE DEBUG RELEASE DEBUG RELEASE DEBUG RELEASE DEBUG RELEASE DEBUG RELEASE DEBUG
            PREFIX_LIST Empty Empty Empty Empty "lib" "lib" "lib" "lib" "lib" "lib" "lib"
            POSTFIX_LIST Empty "_debug" "_static" "_static_debug" Empty "d" Empty "d" Empty "_debug" Empty "_debug"
            EXTENSION_LIST "lib" "lib" "lib" "lib" "so" "so" "a" "a" "dll.a" "a" "dll.a" "a"
            LIBS_LIST "shlwapi.lib" "shlwapi.lib" "shlwapi.lib" "shlwapi.lib" Empty Empty Empty Empty Empty Empty Empty Empty
            DEFINITIONS_LIST
            "GFLAGS_IS_A_DLL=1"
            "GFLAGS_IS_A_DLL=1"
            "GFLAGS_IS_A_DLL=0"
            "GFLAGS_IS_A_DLL=0"
            "GFLAGS_IS_A_DLL=1"
            "GFLAGS_IS_A_DLL=1"
            "GFLAGS_IS_A_DLL=0"
            "GFLAGS_IS_A_DLL=0"
            "GFLAGS_IS_A_DLL=1"
            "GFLAGS_IS_A_DLL=1"
            "GFLAGS_IS_A_DLL=0"
            "GFLAGS_IS_A_DLL=0"
        )

        # make sure dir existed
        touch_fold(${module}_include)
        touch_fold(${module}_lib_dir)

        if(${module}_need_nothreads)
            add_library(_${module}_nothreads UNKNOWN IMPORTED)
            set_target_properties(
                _${module}_nothreads
                PROPERTIES
                IMPORTED_LOCATION "${${module}_lib_dir}/${prefix}gflags_nothreads${postfix}.${extension}"
            )
            add_dependencies(_${module}_nothreads ext_${module})
        else()
            add_library(_${module}_nothreads INTERFACE)
        endif()
        #include(CMakeFindDependencyMacro)
        #find_dependency(Threads)

        add_library(_${module}_thread UNKNOWN IMPORTED)
        set_target_properties(
            _${module}_thread
            PROPERTIES
            IMPORTED_LOCATION "${${module}_lib_dir}/${prefix}gflags${postfix}.${extension}"
        )
        add_dependencies(_${module}_thread ext_${module})

        add_library(interface_lib${module} INTERFACE)
        set_target_properties(interface_lib${module}
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${${module}_include_dir}"
            INTERFACE_LINK_DIRECTORIES "${${module}_lib_dir}"
            INTERFACE_LINK_LIBRARIES "_${module}_thread;_${module}_nothreads;${libs}"
            INTERFACE_COMPILE_DEFINITIONS "${definitions}"
        )
        add_dependencies(interface_lib${module} _${module}_nothreads _${module}_thread)
    endif()
    set(${module}_target_name interface_lib${module} PARENT_SCOPE)
endfunction(gflags_target)