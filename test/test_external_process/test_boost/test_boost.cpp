#include <boost/filesystem/path.hpp>
#include <boost/chrono.hpp>
#include <boost/thread/mutex.hpp>
#include <iostream>

#include <iostream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <map>

using namespace std;
namespace bg = boost::geometry;

const static std::string model_path{""};
const static float conf{.0};
const static std::map<int, std::string> target_class_set({
    {0, ""},
});

double returnIOU(const vector<pair<double, double>>& normalPolygon, const vector<double>& rectangleTwoPoints) {
    typedef bg::model::d2::point_xy<double> point_type;
    typedef bg::model::polygon<point_type> polygon_type;

    polygon_type bbox1, bbox2;
    for (const auto& point : normalPolygon) {
        bg::append(bbox1, point_type(point.first, point.second));
    }

    bg::append(bbox2, point_type(rectangleTwoPoints[0], rectangleTwoPoints[1]));
    bg::append(bbox2, point_type(rectangleTwoPoints[2], rectangleTwoPoints[1]));
    bg::append(bbox2, point_type(rectangleTwoPoints[2], rectangleTwoPoints[3]));
    bg::append(bbox2, point_type(rectangleTwoPoints[0], rectangleTwoPoints[3]));

    polygon_type bbox1Poly = bbox1, bbox2Poly = bbox2;

    std::vector<polygon_type> unionPoly;
    bg::union_(bbox1Poly, bbox2Poly, unionPoly);

    double iou = 0.0;
    if (!bg::intersects(bbox1Poly, bbox2Poly)) {
        iou = 0.0;
    } else {
        std::vector<polygon_type> interPolygon;
        assert(bg::intersection(bbox1Poly, bbox2Poly, interPolygon));
        float interArea = .0;
        for (auto ip: interPolygon){
            interArea += bg::area(ip);
        }
        float unionArea = .0;
        for (auto up: unionPoly){
            unionArea += bg::area(up);
        }
        if (unionArea > 0.0) {
            iou = interArea / unionArea;
        }
    }
    return iou;
};

double calculateIOU(const vector<vector<double>>& boxByUser, const vector<double>& bboxYolo, const std::vector<std::string>& type) {
    vector<double> iouList;
    #pragma omp parallel for
    for (size_t i = 0; i < boxByUser.size(); i++) {
        if (type[i] == "rectangle") {
            if (boxByUser[i].size() != 4) {
                throw runtime_error("rectangle: region format error");
            }
            vector<pair<double, double>> rectangleI = {
                {boxByUser[i][0], boxByUser[i][1]},
                {boxByUser[i][2], boxByUser[i][1]},
                {boxByUser[i][2], boxByUser[i][3]},
                {boxByUser[i][0], boxByUser[i][3]}
            };
            double rectangleIOU = returnIOU(rectangleI, bboxYolo);
            if (rectangleIOU >= 0.0001) {
                iouList.push_back(rectangleIOU);
            }
        } else if (type[i] == "polygon") {
            if (boxByUser[i].size() < 6) {
                throw runtime_error("polygon: region format error");
            }
            size_t polygonINum = boxByUser[i].size();
            if (polygonINum % 2 != 0) {
                throw runtime_error("polygon: region format error");
            }
            vector<pair<double, double>> polygonI;
            for (size_t w = 0; w < polygonINum / 2; w++) {
                polygonI.push_back({boxByUser[i][2 * w], boxByUser[i][2 * w + 1]});
            }
            double polygonIOU = returnIOU(polygonI, bboxYolo);
            if (polygonIOU >= 0.0001) {
                iouList.push_back(polygonIOU);
            }
        } else {
            throw runtime_error("unsupported format");
        }
    }
    if (iouList.size() >= 1) {
        return 1.0;
    } else {
        return 0.0;
    }
};

int main(int argc, char** argv){
    char* t = "/usr/";
    auto k = boost::filesystem::path(t);

    //std::cout << boost::chrono::steady_clock::now() << boost::chrono::system_clock::now() << std::endl;
    return 0;
}