#include <boost/filesystem/path.hpp>
int main(int argc, char** argv){
    char* t = "/usr/";
    auto k = boost::filesystem::path(t);
    return 0;
}