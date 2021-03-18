cd ./build-debug
rm -rf *
cmake ../ -DCMAKE_BUILD_TYPE=Debug -Dvar_external_install_path=../install-debug -Dvar_external_download_dir=../external