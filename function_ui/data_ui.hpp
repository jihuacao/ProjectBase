#ifndef PROJECT_BASE_FUNCTION_UI_DATA_UI_H
#define PROJECT_BASE_FUNCTION_UI_DATA_UI_H
#include <ProjectBase/function_ui/Define.hpp>
#include <ProjectBase/function/data.hpp>
#include <boost/thread/mutex.hpp>

namespace ProjectBase{
    namespace function_ui{
        class PROJECT_BASE_FUNCTION_UI_SYMBOL DataUI{
            public:
                DataUI();
            public:
                void set_data(const ProjectBase::function::Data* const data);
            public:
                void response_immediately(bool on);
                void view_change(const ProjectBase::function::Data& data);
                const ProjectBase::function::Data& show_data();
            public:
                ProjectBase::function::Data::_get_data_func _gf;
                ProjectBase::function::Data::_set_data_func _sf;
        };
    }
}
#endif // ! PROJECT_BASE_FUNCTION_UI_DATA_UI_H