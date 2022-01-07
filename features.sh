usage(){
  echo "
Usage:
    --generator <generator> **search cmake-generators in cmake documentation**
        \"Visual Studio <compiler version> <ide version>\" for Visual Studio compiler
        \"MinGW Makefiles\" for MinGW
    --compiler <the compiler>
    --compiler_path <the path of the compiler>
    --build_type the configuration: [Debug(Default)|Release|RelWithDebInfo|MinSizeRel]
    --build_dir <the dir>
    -h, --help    display this help and exit
"
}
ARGS=`getopt \
    -o h \
    --long help,compiler:,compiler_path:,generator: \
    -n 'feature.bash' -- "$@"`
if [ $? != 0 ]; then echo "Terminating..." ; exit 1 ; fi
eval set -- "${ARGS}"
while true ; do
    case "$1" in
        --genertator) echo "generator $2"; shift 2 ;;
        --compiler) echo "compiler $2"; shift 2 ;;
        --compiler_path) echo "compiler_path $2"; shift 2 ;;
        --) shift ; break ;;
        -h|--help) usage ; exit ;;
        *) echo "Internal error!" ; exit 1 ;;
    esac
done
if ! test ${project_base_build_type}; then project_base_build_type="Debug"; fi
if ! test ${generator}; then echo "you should specify the cmake generator..."; exit 1; fi
if ! test ${compiler}; then echo "using default compiler"; fi

echo "cmake generator: ${generator}"
echo "compiler: ${compiler}"
echo "project_base_build_type: ${project_base_build_type}"