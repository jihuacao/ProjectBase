# this file set the variable for the ProjectBase
#

###################################################################
# set the ProjectBase_config
# ProjectBase_config_dir:
#       the dir where the config files locate
# boost config file:
#       ProjectBase_boost_componment_config_file_name: the name of componment config file for boost componment
#       ProjectBase_boost_componment_config_file_rpath: the rpaht of componement config file for boost componment
###################################################################
get_filename_component(ProjectBase_config_dir ${CMAKE_CURRENT_LIST_DIR} PATH)
set(ProjectBase_config_dir ${ProjectBase_config_dir}/config)
set(ProjectBase_boost_componment_config_file_name boost_componment_config)
set(ProjectBase_boost_componment_config_file_rpath ${ProjectBase_config_dir}/${ProjectBase_boost_componment_config_file_name})


###################################################################
# set the ProjectBase_script
# ProjectBase_script_dir:
#       the dir where the scripts locate
# componment config file processing script:
#       ProjectBase_script_componment_config_file_fix: the rpath of the componment config file processing script
###################################################################
get_filename_component(ProjectBase_script_dir ${CMAKE_CURRENT_LIST_DIR} PATH)
set(ProjectBase_script_dir ${ProjectBase_script_dir}/script)
set(ProjectBase_script_componment_config_file_fix ${ProjectBase_script_dir}/componment_config_file_fix.py)