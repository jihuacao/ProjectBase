cd ./build
#rm -rf CMakeCache.txt
rm -rf *
cmake ../ -DCMAKE_BUILD_TYPE=Debug -Dvar_external_install_path=../install -Dvar_external_download_dir=../external -DCMAKE_INSTALL_PREFIX=../install
