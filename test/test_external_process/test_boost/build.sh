: << the_debug
cmake \
-DCMAKE_BUILD_TYPE=Debug \
-DBUILD_SHARED_LIBS=ON \
-Dexternal_build_shared=OFF \
-Dvar_external_root_dir=/data/caojihua/Project/ProjectBase/tools/external/ \
-Dexternal_build_type=RELEASE \
-Dboost_build_type=RELEASE \
--log-level=DEBUG \
-S $(dirname $(pwd)) \
-B $(pwd) \
-Dboost_version=1.81.0
the_debug

:<< the_release
cmake \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_SHARED_LIBS=ON \
-Dexternal_build_shared=OFF \
-Dvar_external_root_dir=/data/caojihua/Project/ProjectBase/tools/external/ \
-Dexternal_build_type=RELEASE \
-Dboost_build_type=RELEASE \
--log-level=DEBUG \
-S $(dirname $(pwd)) \
-B $(pwd) \
-Dboost_version=1.81.0
the_release