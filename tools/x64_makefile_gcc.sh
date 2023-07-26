call_dir=$(pwd)
bash_dir=$(cd $(dirname $0); pwd)
workroot=$(dirname $(cd $(dirname $0); pwd))

Platform=x64
Configuration=DEBUG
BuildDir=${call_dir}/BinaryOut/${Platform}-${Configuration}
mkdir -p ${BuildDir}
InstallDir=${call_dir}/InstallOut/${Platform}-${Configuration}
mkdir -p ${InstallDir}

source ${bash_dir}/build.sh \
--build_dir=${BuildDir} \
--platform="x64" \
--configuration=${Configuration} \
--generator="Unix Makefiles" \
--compiler=gcc \
--build_share \
--linker=ld \
--build-options=" -DCMAKE_INSTALL_PREFIX=${InstallDir}"