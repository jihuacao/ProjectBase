usage(){
    echo "help message"
}
ARGS=`getopt \
    -o hab:c:: \
    --long help,along,blong:,clong:: \
    -n 'example.bash' -- "$@"`
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi
eval set -- "${ARGS}"
while true ; do # 对与参数字符的遍历不结束，该项永远为true
    case "$1" in
        ## 由于a是属于无参数值类型，因此只需要shift 1
        -a|--along) echo "Option $1"; shift 1;;
        ## 由于b，b-long是有参数值类型，则需要shift 2，使用$2来获得值
        -b|--blong) echo "Option $1, argument $2"; shift 2;;
        -c|--clong)
            ## 由于c，c-long是一个option类型的参数，因此需要使用case进行强约束，避免混乱
            case "$2" in
                "") echo "Option $1, no argument"; shift 2;;
                *)  echo "Option $1, argument $2" ; shift 2;;
            esac ;;
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
