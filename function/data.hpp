#ifndef PROJECT_BASE_FUNCTION_DATA_H
#define PROJECT_BASE_FUNCTION_DATA_H
#include <ProjectBase/function/Define.hpp>
namespace ProjectBase{
    namespace function{
        class PROJECT_BASE_FUNCTION_SYMBOL Data{
            public:
                typedef const ProjectBase::function::Data& (*_get_data_func)();
                typedef bool (*_set_data_func)(const ProjectBase::function::Data& data);
            public:
                Data();
                Data(const Data& other);
                Data(Data&& ref);
            public:
                
        };
    }
}
#endif // ! PROJECT_BASE_FUNCTION_DATA_H