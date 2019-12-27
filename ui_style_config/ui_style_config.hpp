#ifndef PROJECT_BASE_UI_STYLE_CONFIG_UI_STYLE_CONFIG_H
#define PROJECT_BASE_UI_STYLE_CONFIG_UI_STYLE_CONFIG_H
#include <ProjectBase/ui_style_config/Define.hpp>
#include <boost/shared_ptr.hpp>
namespace ProjectBase{
    namespace ui_style_config{
        class ui_style_config;
        class PROJECT_BASE_UI_STYLE_CONFIG_SYMBOL UIStyleConfig{
            public:
                typedef boost::shared_ptr<ui_style_config> _implement;
            public:
                UIStyleConfig();
            public:
                typedef void (*_widget_process_func)(QWidget* ptr);
                void ProcessWidget();
            private:
                ProjectBase::ui_style_config::UIStyleConfig::_implement _impl;
        };
    }
}
#endif // ! PROJECT_BASE_UI_STYLE_CONFIG_UI_STYLE_CONFIG_H