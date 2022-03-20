#ifdef PROJECT_BASE_UTIL_CRASH_HANDLE_H
#else
#define PROJECT_BASE_UTIL_CRASH_HANDLE_H
#include <ProjectBase/util/Define.hpp>
namespace ProjectBase{
    namespace Util{
        class PROJECT_BASE_UTIL_SYMBOL CrashHandler{
            public:
                CrashHandler();
        };
    };
}
#ifdef __GNUC__
#else
#endif
#endif