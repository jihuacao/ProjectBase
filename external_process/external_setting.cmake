set(external_install_path ${CMAKE_CURRENT_BINARY_DIR}/install CACHE STRING "the path for installing the external")
option(external_build_shared "the lib link type for building the external" ON)
set(external_download_dir ${CMAKE_CURRENT_BINARY_DIR} CACHE STRING "the dir for download the source of the third party repo")