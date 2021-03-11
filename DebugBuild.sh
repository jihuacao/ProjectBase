cd ./build-debug
rm -rf *
cmake ../ -DCMAKE_BUILD_TYPE=Debug -Dexternal_install_path=../install-debug -Dexternal_download_dir=../external