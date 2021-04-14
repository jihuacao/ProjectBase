cd ./build-debug
rm -rf CMakeCache.txt
cmake ../ -DCMAKE_BUILD_TYPE=Debug -Dvar_external_install_path=../install-debug -Dvar_external_download_dir=../external