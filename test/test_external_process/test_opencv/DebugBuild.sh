cd ./build
#rm -rf CMakeCache.txt
rm -r *
cmake \
-DCMAKE_BUILD_TYPE=Debug \
-Dvar_external_root_dir=$(echo $(cd ../../../../tools/external/ && pwd)) \
-Dexternal_build_shared=ON \
-Dprotobuf_build_type=DEBUG \
../
