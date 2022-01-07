macro(project_base_system_message)
    message(STATUS "your system name is ${CMAKE_SYSTEM_NAME}[CMAKE_SYSTEM_NAME]--${CMAKE_CURRENT_LIST_FILE}")
    message(STATUS "cmake generator: ${CMAKE_GENERATOR}--${CMAKE_CURRENT_LIST_FILE}")
    message(STATUS "cmake generator platform: ${CMAKE_GENERATOR_PLATFORM}--${CMAKE_CURRENT_LIST_FILE}")
    message(STATUS "cmake generator toolset: ${CMAKE_GENERATOR_TOOLSET}--${CMAKE_CURRENT_LIST_FILE}")
    message(STATUS "cmake generator instance: ${CMAKE_GENERATOR_INSTANCE}--${CMAKE_CURRENT_LIST_FILE}")
endmacro(project_base_system_message)

macro(cmakelists_base_header)
    message(STATUS ${CMAKE_CURRENT_LIST_FILE})
endmacro(cmakelists_base_header)

macro(show_all_variables)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endmacro(show_all_variables)

#####################################################################
#####################################################################
macro(display_properties_supported)
    # Get all propreties that cmake supports
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
    message (STATUS CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST})
endmacro(display_properties_supported)

#####################################################################
#####################################################################
macro(show_target_properties target)
    # Get all propreties that cmake supports
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)

    # Convert command output into a CMake list
    STRING(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    STRING(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")

    function(print_target_properties tgt)
        if(NOT TARGET ${tgt})
          message(FATAL_ERROR "There is no target named '${tgt}'")
          return()
        endif()

        foreach (prop ${CMAKE_PROPERTY_LIST})
            string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" prop ${prop})

            if(prop STREQUAL "LOCATION" OR prop MATCHES "^LOCATION_" OR prop MATCHES "_LOCATION$")
                continue()
            endif()

            get_property(propval TARGET ${tgt} PROPERTY ${prop} SET)
            if (propval)
                get_target_property(propval ${tgt} ${prop})
                message (STATUS "${tgt} ${prop} = ${propval}")
            endif()
        endforeach(prop)
    endfunction(print_target_properties)

    print_target_properties(${target})
endmacro(show_target_properties)


#####################################################################
# debug_the_src_with_abspath :
#   在构建target时，我们会使用file(GLOB ***)（虽然不能支持新obj更新响应），
#   我们会获得一个list，我们希望能够输出该list的信息，但是一般该list是abspath，
#   我们可以使用该function，实现list的debug信息
#####################################################################
function(debug_the_src_with_abspath abspaths)
    list(APPEND group)
    #message(STATUS "debug: asdasd: ${group}")
    foreach(item ${${abspaths}})
        get_filename_component(parent ${item} DIRECTORY)
        get_filename_component(file ${item} NAME)
        if(parent IN_LIST group)
        else()
            #message(STATUS "debug: asdasdsad:add")
            #message(STATUS "debug: T:${parent} TT:${group}")
            list(APPEND group ${parent})
        endif()
        list(APPEND ${parent} ${file})
    endforeach(item)
    set(msg "")
    #message(STATUS "debug: ASDASD: ${group}")
    foreach(g ${group})
        set(msg "${msg}:|parent_dir(${g}):${${g}}")
    endforeach(g)
    message(DEBUG "${msg}")
endfunction(debug_the_src_with_abspath)