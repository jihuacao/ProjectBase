message(${CMAKE_CURRENT_LIST_FILE})
set(with_boost ON)
include(popular_message)
project_base_system_message()
include(ExternalProject)

include(external_setting)

set(module boost)
list(
    APPEND 
    ${module}_all_version 
    "1.71.0"
    )

set(
    ${module}_supported_url 
    https://dl.bintray.com/boostorg/release/1.71.0/source/boost_1_71_0.tar.gz
    )

set(
    ${module}_supported_hash
    96b34f7468f26a141f6020efb813f1a2f3dfb9797ecf76a7d7cbd843cc95f5bd
    )

version_selector(${module} ${module}_all_version "1.71.0")
version_url_hash_match(${module} ${module}_all_version ${module}_supported_url ${module}_supported_hash ${module}_version)