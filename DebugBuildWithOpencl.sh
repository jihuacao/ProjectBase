cd ./build-debug
make clean
rm -f CMakeCache.txt
rm -f Makefile
cmake ../ -DCMAKE_INSTALL_PREFIX=../install-debug -DCMAKE_BUILD_TYPE=Debug -Dvar_external_install_path=../install-debug -Dvar_external_download_dir=../external -DOPENCL=ON