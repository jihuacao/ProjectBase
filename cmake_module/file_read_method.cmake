####################################################################
# this macro provide read componment config from file, and save the 
# needed componment to ${${name_prefix}}_with:list
# unneeded componment to ${${name_prefix}}_without:list
# file_path:
#       the componment config file rpath, for example, config/boost_componment_config
#       the format should be: one componment in one line and ON represent needing componment
#       OFF represent do not need componment
# name_prefix:
#       the package prefix, for example: set name_prefix Boost for boost library
####################################################################
macro(componment_config_read file_path name_prefix)
    file(STRINGS ${${file_path}} ConfigContents)
    set(${name_prefix}_with)
    set(${name_prefix}_without)
    foreach(NameAndValue ${ConfigContents})
        # Strip leading spaces
        string(REGEX REPLACE "^[ ]+" "" NameAndValue ${NameAndValue})
        # Find variable name
        string(REGEX MATCH "^[^=]+" Name ${NameAndValue})
        # Find the value
        string(REPLACE "${Name}=" "" Value ${NameAndValue})
        # Set the variable
        if(${Value})
            list(APPEND ${name_prefix}_with ${Name})
        else()
            list(APPEND ${${name_prefix}}_without ${Name})
        endif()
    endforeach()
endmacro(componment_config_read)