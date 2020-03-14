#ifndef PROJECT_BASE_FUNCTION_DATA_CONTAINER_H
#define PROJECT_BASE_FUNCTION_DATA_CONTAINER_H
#include <string>
#include <ProjectBase/function/Define.hpp>
#include <ProjectBase/data/data.hpp>
#include <ProjectBase/function/name_container.hpp>

namespace ProjectBase{
    namespace function{
        class PROJECT_BASE_FUNCTION_SYMBOL DataContainer{
            public:
                typedef std::string data_name;
            public:
                DataContainer();
                DataContainer(const DataContainer& other);
                DataContainer(DataContainer&& ref);
            public:
               const Data::_get_data_func get_data_func(const data_name& target) const;
               const Data::_get_data_func get_data_func(data_name&& target) const;
               const ProjectBase::function::Data& get_data(const data_name& target) const;
               const ProjectBase::function::Data& get_data(data_name&& target) const;
               const Data::_set_data_func set_data_func(const data_name& target) const;
               const Data::_set_data_func set_data_func(data_name&& target) const;
            public:
               const void register_data(const ProjectBase::function::Data& data);
            protected:
        };
    }
}
#endif // !PROJECT_BASE_FUNCTION_DATA_CONTAINER_H