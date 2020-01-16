#ifndef PROJECT_BASE_FUNCTION_NAME_CONTAINER_H
#define PROJECT_BASE_FUNCTION_NAME_CONTAINER_H
#include <ProjectBase/function/Define.hpp>
#include <ProjectBase/function/name.hpp>
namespace ProjectBase{
    namespace function{
        class PROJECT_BASE_FUNCTION_SYMBOL NameContainer{
            public:
                NameContainer();
                NameContainer(const NameContainer& other);
                NameContainer(NameContainer&& ref);
            public:
        };
    }
}
#endif // ! PROJECT_BASE_FUNCTION_NAME_CONTAINER_H