#ifndef PROJECT_BASE_PLUGIN_MANAGER_INNER_PLUGIN_MANAGER_H
#define PROJECT_BASE_PLUGIN_MANAGER_INNER_PLUGIN_MANAGER_H
#include <ProjectBase/plugin_manager/plugin_manager.hpp>
#include <boost/property_tree/ptree.hpp>

namespace ProjectBase{
    namespace PluginManager{
        class plugin_cell{

        };

        class plugin_registry{
            public:
                plugin_registry();
            public:
            
            private:
                boost::property_tree::basic_ptree<std::string, std::string, std::string> _tree;
        };

        class plugin_manager{
            public:
                plugin_manager();
            public:
                
        };
    }
}
#endif