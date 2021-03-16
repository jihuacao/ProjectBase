#include <ProjectBase/file_system/Define.hpp>
#include <boost/filesystem.hpp>

namespace ProjectBase{
    namespace file_system{
        /**
         * \brief brief
         * \note note
         * \author none
         * \param[in] in
         * \param[out] out
         * \return return
         * \retval retval
         * \since version
         * */
        inline bool is_path(){
            return boost::filesystem::exists("a.txt");
        };
        /**
         * \brief brief
         * \note note
         * \author none
         * \param[in] in
         * \param[out] out
         * \return return
         * \retval retval
         * \since version
         * */
        inline bool is_file(){
            return true;
        };
        /**
         * \brief brief
         * \note note
         * \author none
         * \param[in] in
         * \param[out] out
         * \return return
         * \retval retval
         * \since version
         * */
        inline bool is_fold(){
            return true;
        };
        /**
         * \brief brief
         * \note note
         * \author none
         * \param[in] in
         * \param[out] out
         * \return return
         * \retval retval
         * \since version
         * */
        inline bool path_exist(){
            return true;
        };
        /**
         * \brief brief
         * \note note
         * \author none
         * \param[in] in
         * \param[out] out
         * \return return
         * \retval retval
         * \since version
         * */
        inline bool file_exist(){
            return true;
        };
        inline bool fold_exist(){
            return true;
        };
    }
}