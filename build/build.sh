usage(){
    echo "help message:"
    echo "--log-level=[DEBUG|RELEASE]"
}
ARGS=`getopt \
    -o hab:c:: \
    --long help,along,blong:,component::,log-level:: \
    -n 'example.bash' -- "$@"`
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi
eval set -- "${ARGS}"
while true ; do
    case "$1" in
        -a|--along) echo "Option $1"; shift 1;;
        -b|--blong) echo "Option $1, argument $2"; shift 2;;
        -c|--component)
            case "$2" in
                "") echo "Option $1, no argument"; shift 2;;
                *)  echo "Option $1, argument $2" ; shift 2;;
            esac ;;
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
-Dvar_external_root_dir="${exec_dir}/external" \
--log-level=${log_level} \
-LAH \
> ${exec_dir}/cmake_config.log
cp ${exec_dir}/cmake_config.log ${output_path}
cp ${exec_dir}/CMakeCache.txt ${output_path}
