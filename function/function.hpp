#ifndef PROJECT_BASE_FUNCTION_FUNCTION_H
#define PROJECT_BASE_FUNCTION_FUNCTION_H
#include <string>
#include <ProjectBase/function/Define.hpp>
#include <ProjectBase/data/data.hpp>
#include <ProjectBase/data/data_container.hpp>

namespace ProjectBase{
    namespace function{
        typedef std::string FUNCTION_NAME;
        class PROJECT_BASE_FUNCTION_SYMBOL Function
        {
            public:
                Function(/* args */);
                ~Function();
            public:
                ProjectBase::function::DataContainer& data_container() const;
                void executa() const;
            protected:
                /* data */
        };
    }
}
#endif // !1PROJECT_BASE_FUNCTION_H