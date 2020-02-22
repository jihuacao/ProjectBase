################################################################
# 
################################################################
function(touch_fold fold_dir)
    if(EXISTS ${${fold_dir}})
    else()
        execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${${fold_dir}})
    endif()
endfunction(touch_fold)