#ifndef PROJECT_BASE_FUNCTION_NAME_H
#define PROJECT_BASE_FUNCTION_NAME_H
#include <boost/container/string.hpp>
#include <ProjectBase/function/Define.hpp>
namespace ProjectBase{
    namespace function{
        class _inner_name;
        class PROJECT_BASE_FUNCTION_SYMBOL Name{
            public:
                Name(const boost::container::string& str);
                Name(boost::container::string&& str);
            public:
                operator boost::container::string() const;
            protected:
                _inner_name* _impl;
        };
    }
}
#endif // ! PROJECT_BASE_FUNCTION_NAME_H