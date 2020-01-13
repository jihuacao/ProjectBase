################################################################
# this fucntion make sure the file exist, and generate a target name
################################################################
function(touch_file touch_target generated_target_posefix url hash dir file_name)
    find_file(found_download_file ${${file_name}} ${${dir}})
    set(actual_download_command axel -k -n 10 -av ${${url}} -o ${${dir}}/${${file_name}})
    if(${found_download_file} STREQUAL "found_download_file-NOTFOUND")
        set(download_command ${actual_download_command})
    else()
        file(SHA256 ${${dir}}/${${file_name}} exist_sha)
        if(${exist_sha} STREQUAL ${${hash}})
            message(STATUS "${${file_name}} found in ${${dir}}")
            set(download_command )
        else()
            set(download_command ${actual_download_command})
        endif()
    endif()

    add_custom_target(touch_${${generated_target_posefix}} ALL COMMAND ${download_command})
    set(${touch_target} touch_${${generated_target_posefix}} PARENT_SCOPE)
endfunction(touch_file)