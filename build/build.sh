usage(){
    echo "help message"
}
ARGS=`getopt \
    -o hab:c:: \
    --long help,along,blong:,component:: \
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
