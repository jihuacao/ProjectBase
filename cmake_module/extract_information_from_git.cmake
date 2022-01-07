function(git_current_tag tag)
    execute_process(
        COMMAND git describe --abbrev=0
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        TIMEOUT 5
        OUTPUT_VARIABLE GIT_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(${tag} ${GIT_TAG} PARENT_SCOPE)
endfunction(git_current_tag)

function(git_current_commit commit)
    execute_process(
        COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        TIMEOUT 5
        OUTPUT_VARIABLE GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(${commit} ${GIT_COMMIT} PARENT_SCOPE)
endfunction(git_current_commit)