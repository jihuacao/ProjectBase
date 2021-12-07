#[[
    function_name: 
]]
function(version_from_git major minor maintenance build)
    include(cmake/extract_information_from_git.cmake)
    git_current_tag(git_tag)
    git_current_commit(git_commit)
    string(REPLACE "." ";" a ${git_tag})
    list(LENGTH a len)
    message(STATUS "len:${len}")
    list(GET a 0 get)
    set(${major} ${get} PARENT_SCOPE)
    list(GET a 1 get)
    set(${minor} ${get} PARENT_SCOPE)
    list(GET a 2 get)
    set(${maintenance} ${get} PARENT_SCOPE)
    set(${build} ${git_commit} PARENT_SCOPE)
endfunction(version_from_git)