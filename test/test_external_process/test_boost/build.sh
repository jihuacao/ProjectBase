cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=ON \
-Dexternal_build_shared=OFF \
-Dvar_external_root_dir=/data/caojihua/Project/ProjectBase/tools/external/ \
--log-level=DEBUG \
-S $(dirname $(pwd)) \
-B $(pwd) \
-Dboost_version=1.81.0