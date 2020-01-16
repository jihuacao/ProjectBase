#ifndef PROJECT_BASE_FUNCTION_FUNCTION_H
#define PROJECT_BASE_FUNCTION_FUNCTION_H
#include <ProjectBase/function/Define.hpp>
#include <ProjectBase/function/data.hpp>
#include <ProjectBase/function/data_container.hpp>
namespace ProjectBase{
    namespace function{
        class PROJECT_BASE_FUNCTION_SYMBOL Function
        {
            public:
                Function(/* args */);
                ~Function();
            public:
                ProjectBase::function::DataContainer& data_container() const;
                void run() const; 
            protected:
                /* data */
        };
    }
}
#endif // !1PROJECT_BASE_FUNCTION_H