root_dir=$($(cd $(dirname $0)); pwd)
cd ${root_dir}
echo ${root_dir}
source ${root_dir}/features.sh

#cd ./build-debug
#rm CMakeCache.txt
#rm Makefile
#cmake ../ \
#-DCMAKE_INSTALL_PREFIX=../install-debug \
#-DCMAKE_BUILD_TYPE=Debug \
#-Dvar_external_install_path=../install-debug \
#-Dvar_external_download_dir=../external