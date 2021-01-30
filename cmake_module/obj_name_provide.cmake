##############################################################
# this macro provide a method to get the target name base on the build_type, shared_or_not and obj_base_name
# build_type: the type of the building, Release, Debug, MinSizeRel or RelWithDebInfo, generate the ${type}
# shared_or_not: the bool, ON: the target is shared and OFF: the target is static, generate the ${opt} and ${opt_type}
# obj_base_name: the base name of the obj, which would be
##############################################################
macro(obj_name_provide build_type shared_or_not obj_base_name)
    # generate target name
    string(TOUPPER ${${build_type}} _build_type)
    if("${_build_type}" STREQUAL "RELEASE")
        set(type)
    elseif("${_build_type}" STREQUAL "DEBUG")
        set(type d)
    elseif("${_build_type}" STREQUAL "MINSIZEREL")
        set(type msr)
    elseif("${_build_type}" STREQUAL "RELWITHDEBINFO")
        set(type rwdi)
    else()
        message(FATAL_ERROR "unsupported build_type : ${build_type}")
    endif()
    unset(_build_type)

    if(${${shared_or_not}})
        set(opt)
        set(${${obj_base_name}}_build_link_type SHARED)
    elseif(Not${${shared_or_not}})
        set(opt _static)
        set(${${obj_base_name}}_build_link_type STATIC)
    endif()

    set(${${obj_base_name}}_target_name ${${obj_base_name}}${type}${opt} CACHE INTERNAL "module target name")
endmacro(obj_name_provide)

macro(executable_name_provide build_type obj_base_name)
    string(TOUPPER ${${build_type}} _build_type)
    if("${_build_type}" STREQUAL "RELEASE")
        set(type)
    elseif("${_build_type}" STREQUAL "DEBUG")
        set(type d)
    elseif("${_build_type}" STREQUAL "MINSIZEREL")
        set(type msr)
    elseif("${_build_type}" STREQUAL "RELWITHDEBINFO")
        set(type rwdi)
    else()
        message(FATAL_ERROR "unsupported build_type : ${build_type}")
    endif()
    unset(_build_type)
    #set(${${obj_base_name}}_target_name ${${obj_base_name}}${type})
    set(${${obj_base_name}}_target_name ${${obj_base_name}}${type} CACHE INTERNAL "module target name")
endmacro(executable_name_provide)