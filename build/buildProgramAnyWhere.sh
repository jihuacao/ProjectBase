exec_dir=$(pwd)
shell_dir=$(dirname $0)
shell_path=$(cd $(dirname $0); pwd)
output_path=${shell_dir}/build_cache
mkdir ${output_path}
$(cd ${exec_dir})

cmake \
-DCMAKE_BUILD_TYPE=DEBUG \
-A x64 \
-G "Visual Studio 16 2019" \
-S "$(dirname ${shell_dir})" \
-B "${exec_dir}" \
-Dexternal_build_shared=OFF \
--log-level=DEBUG \
-LAH \
> ${exec_dir}/cmake_config.log
cp ${exec_dir}/cmake_config.log ${output_path}
cp ${exec_dir}/CMakeCache.txt ${output_path}