/****************************************************************************
<plugin_manager.hpp>
Copyright (C) <2020/5/14>  <name of author>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

*****************************************************************************/
#ifndef PROJECT_BASE_PLUGIN_MANAGER_PLUGIN_MANAGER_H
#define PROJECT_BASE_PLUGIN_MANAGER_PLUGIN_MANAGER_H
#include <ProjectBase/plugin_manager/Define.hpp>
#include <boost/container/string.hpp>
#include <boost/filesystem/path.hpp>

namespace ProjectBase{
    namespace PluginManager{
        class plugin_registry;
        class PluginRegistry{
            public:
                PluginRegistry();
            public:
                /*
                this function
                */
                void append(const boost::container::string& name, const boost::filesystem::path& path);
                /*
                */
                void remove(const boost::container::string& name);
                /**/
                void check();
                /**/
                boost::container::string get();
                /**/
            private:
                ProjectBase::PluginManager::plugin_registry* _impl;
        };

        class plugin_manager;
        class PluginManager{
            public:
                PluginManager();
            public:
                const PluginRegistry& registry() const;               
            private:
                ProjectBase::PluginManager::plugin_manager* _impl;
        };   
    }
}
#endif