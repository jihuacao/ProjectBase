exec_dir=$(pwd)
bash_dir=$(cd $(dirname $0); pwd)

usage(){
    echo "usage: "
    echo "--output_root: the test result would be save to this directory"
    echo "--exclude: 过滤一些模块, 可重复指定，也可通过空格来代表list
    --exclude=\"a b c\"
    \"\*\" 表示所有都去除"
    echo "--include: 指定必须包含的模块"
}

ARGS=`getopt \
    -o h \
    --long help,output_root::, \
    --long exclude::, \
    --long include::, \
    -n 'example.bash' -- "$@"`
if [ $? != 0 ] ; then echo "Terminating..." >&2; exit 1; fi
eval set -- "${ARGS}"
output_root=${exec_dir}/test_external_process
while true; do
    case "$1" in 
        -h|--help) usage; exit 1;;
        --output_root) echo "output_root: ${2}"; output_root=$2; shift 2;;
        --exclude) echo "exclude: ${2}"; exclude="${exclude[@]} ${2}"; shift 2;;
        --include) echo "include: ${2}"; include="${include[@]} ${2}"; shift 2;;
        -) shift 1; break;;
        --) shift 1; break;;
        *) echo "internal error"; exit 1;;
    esac
done
echo "Remaining arguments:"
for arg do
   echo '--> '"\`$arg'" ;
done

_target_folder=($(ls $(dirname ${bash_dir})))
target_folder=()

: '过滤器'
echo ${exclude[@]}
for t in ${_target_folder[@]}; do
    _exclude=0
    for _ex in ${exclude[@]}; do
        if [[ ${t} == ${_ex} || ${_ex} == "\*" ]]; then
            _exclude=1
            break
        fi
    done
    _include=0
    : ' if the target is in the {include}, 
    it should be run, even it is in the exclude '
    for _in in ${include[@]}; do
        if [[ ${t} == ${_in} || ${_in} == "\*" ]]; then
            _include=1
            break
        fi
    done
    if [[ (${_exclude} != 1 || ${_include} == 1) && ${t} != "script" ]]; then
        target_folder=(${target_folder[@]} $t)
    fi
done
echo "run test: ${target_folder[*]}"

: '检查路径安全，防止非法路径下建立文件文件夹以及删除操作'
dir_not_empty_check(){
    if [[ $1 == "" ]]; then
        echo "$2 is empty, it is dangerous"
        exit 1
    fi
}
echo "output root: ${output_root}"
: '需要检查路径安全性'
dir_not_empty_check "${output_root}" "output_root"
mkdir ${output_root}

: '不同平台的自适应'
system=$(uname)
if [[ ${system} == "Linux" ]];then
    echo "Linux"
    generator="Unix Makefiles"
    platform=""
elif [[ ${system} == *"CYGWIN"* ]];then
    echo "WindowsCygwin"
    generator="Unix Makefiles"
    cmake_c_compiler="x86_64-w64-mingw32-gcc.exe"
    cmake_cxx_compiler="x86_64-w64-mingw32-g++.exe"
    platform=""
elif [[ ${system} == *"MINGW"* ]];then
    echo "WindowsGitBash"
    generator="Visual Studio 16 2019"
    platform="x64"
else
    echo "None, get system failed!"
    exit 1
fi

: '不同的target'
for ep_target in ${target_folder[@]}; do
    : 'DEBUG RELEASE'
    echo "run ${ep_target}"
    for configuration in "DEBUG" "RELEASE"; do
        : '共享库 静态库'
        echo "run ${ep_target}.${configuration}"
        for shared in "OFF" "ON"; do
            # if [[ ${configuration} != "RELEASE" || ${shared} != "ON" ]]; then
            #     continue
            # fi
            echo "run ${ep_target}.${configuration}.Shared[${shared}]"
            # echo "rm ${output_root}/external/external_build -rf" && rm ${output_root}/external/external_build -rf
            # echo "rm ${output_root}/external/external_install -rf" && rm ${output_root}/external/external_install -rf
            : '外部依赖未找到 外部依赖找到
            如果依赖未找到，那么就涉及了依赖下载的问题，**可能存在下载失败**'
            for has_dep in "OFF" "ON"; do
                echo "run ${ep_target}.${configuration}.Shared[${shared}].HasDep[${has_dep}]"
                output_dir=${output_root}/${ep_target}_${configuration}_${shared}_${has_dep}
                : '需要检查路径安全性'
                dir_not_empty_check "${output_dir}" output_dir
                mkdir ${output_dir}
                echo "rm ${output_dir}/* -rf" && rm ${output_dir}/* -rf
                : '运行cmake进行配置'
                cmake \
                -A "${platform}" \
                -G "${generator}" \
                -DCMAKE_C_COMPILER="${cmake_c_compiler}" \
                -DCMAKE_CXX_COMPILER="${cmake_cxx_compiler}" \
                -S "$(dirname ${bash_dir})/${ep_target}" \
                -B "${output_dir}" \
                -DCMAKE_BUILD_TYPE="${configuration}" \
                -Dvar_external_root_dir="${output_root}/external" \
                -Dexternal_build_shared="${shared}" \
                --log-level=DEBUG \
                -LAH 2>&1 1>"${output_dir}/cmake_config.log"
                : '运行构建'
                if [[ ${system} == "Linux" ]]; then
                    echo "test for ${system} is not completed"
                elif [[ ${system} == *"MINGW"* ]]; then
                    cmd.exe /C "\"D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat\" x64 \
                    && msbuild -p:configuration=${configuration} -p:platform=x64 ${output_dir}/${ep_target}.sln"
                elif [[ ${system} == *"CYGWIN"* ]]; then
                    cd ${output_dir}
                    make
                else
                    echo "${system} is not defined"
                    exit 1
                fi
                
                : '需要进行是否成功构建的检查'
                # exit 1
            done
        done
    done
    exit 1
done