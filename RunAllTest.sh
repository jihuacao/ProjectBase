bash_dir=$($(cd $(dirname $0)); pwd)
external_root_dir=
binary_dir=
test_arch=
test_compiler=
test_configuration=
test_platform=
test_object_type=
# test external process
mkdir ${external_root_dir}
mkdir ${binary_dir}
# config all
cmake \
-S ${bash_dir} \
-B ${binary_dir} \
-A ${test_arch} \
-G ${test_compiler} \
-Dexternal_root_dir=${external_root_dir} \
-DCMAKE_BUILD_TYPE=${test_configuration} \
# build all
if [[ ]]
# run module test
## test util