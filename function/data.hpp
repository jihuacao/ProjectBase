#ifndef PROJECT_BASE_FUNCTION_DATA_H
#define PROJECT_BASE_FUNCTION_DATA_H
#include <ProjectBase/function/Define.hpp>
namespace ProjectBase{
    namespace function{
        class PROJECT_BASE_FUNCTION_SYMBOL Data{
            public:
                Data();
                Data(const Data& other);
                Data(Data&& ref);
            public:

        };
    }
}
#endif // ! PROJECT_BASE_FUNCTION_DATA_H