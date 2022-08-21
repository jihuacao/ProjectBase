project_root=$(dirname $(cd $(dirname $0); pwd))
bash_dir=$(cd $(dirname $0); pwd)
system=$(uname)
BuildDir=""
Platform=""
Configuration=""
BuildShare=OFF
Generator=""
Compiler=""
Linker=""
usage(){
    echo "help message:"
    echo "--build_dir specify the output path for config"
    echo "--platform specify the platform for config x64|x86|arm"
    echo "--configuration specify the configuration for config DEBUG|RELEASE|..."
    echo "--generator specify the generator for cmake config Visual Studio 16 2019|Makefile|..."
    echo "--compiler specify the compiler path"
    echo "--linker specify the linker path"
    echo "--component specify the component to build"
    echo "--log-level=[DEBUG|RELEASE]"
}
ARGS=`getopt \
    -o hab:: \
    -o c:: \
    --long build_dir:: \
    --long platform:: \
    --long configuration:: \
    --long build_share \
    --long generator:: \
    --long compiler:: \
    --long linker:: \
    --long component:: \
    --long build-options:: \
    --long log-level:: \
    --long help \
    -n 'example.bash' -- "$@"`
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi
eval set -- "${ARGS}"
while true ; do
    case "$1" in
        --build_dir)
            echo "build dir: $2"; BuildDir=$2; shift 2 ;;
        --platform)
            echo "platform: $2"; Platform=$2; shift 2 ;;
        --configuration)
            echo "configuration: $2"; Configuration=$2; shift 2 ;;
        --build_share)
            echo "build with share: $2"; BuildShare=ON; shift 1 ;;
        --generator)
            echo "generator: $2"; Generator=$2; shift 2 ;;
        --compiler)
            echo "compiler: $2"; Compiler=$2; shift 2 ;;
        --linker)
            echo "linker: $2"; Linker=$2; shift 2 ;;
        --component)
            case "$2" in
                "") echo "Option $1, no argument"; shift 2;;
                *)  echo "Option $1, argument $2" ; shift 2;;
            esac ;;
        --build-options)
            echo "build options: $2"; BuildOptions=$2; shift 2 ;;
        --log-level)
            log_level_info(){
                echo "set log level to $1"
            }
            case "$2" in
                "DEBUG") log_level_info $2;;
                "RELEASE") log_level_info $2;;
                *) echo "error log level: $2"; usage; exit 1;;
            esac
            log_level=$2
            shift 2;;
        -h|--help) usage; exit 1;;
        --) shift 1; break;;
        -) shift 1; break;;
        *) echo "Internal error!" ; exit 1 ;;
    esac
done
echo "Remaining arguments:"
for arg do
   echo '--> '"\`$arg'" ;
done

external_source_dir=${bash_dir}/external
mkdir ${external_source_dir}

# 一些系统差异性
system_diff_options=""
if [[ ${system} == "Linux" ]]; then
    system_diff_options=""
else
    system_diff_options="${system_diff_options} -A \"${Platform}\""
fi

#echo " \
cmake \
-DCMAKE_BUILD_TYPE=${Configuration} \
${system_diff_options} \
-G "${Generator}" \
-S "${project_root}" \
-B "${BuildDir}" \
-Dexternal_build_shared=${BuildShare} \
-Dvar_external_root_dir="${external_source_dir}" \
${BuildOptions} \
--log-level=DEBUG \
-LAH \
> ${BuildDir}/cmake_config.log

cp ${BuildDir}/cmake_config.log ${bash_dir}
cp ${BuildDir}/CMakeCache.txt ${bash_dir}