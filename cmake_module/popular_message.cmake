macro(project_base_system_message)
message("your system name is ${CMAKE_SYSTEM_NAME}, using CMAKE_SYSTEM_NAME to get the system name--${CMAKE_CURRENT_LIST_FILE}")
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