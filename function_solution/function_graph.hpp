#ifndef PROJECT_BASE_FUNCTION_SOLUTION_FUNCTION_GRAPH_H
#define PROJECT_BASE_FUNCTION_SOLUTION_FUNCTION_GRAPH_H
#include <ProjectBase/function_solution/Define.hpp>
namespace ProjectBase{
    namespace function_solution{
        class PROJECT_BASE_FUNCTION_SOLUTION_SYMBOL FunctionGraph{
            public:
                FunctionGraph();
                FunctionGraph(FunctionGraph&& ref);
                FunctionGraph(const FunctionGraph& other);
            public:
        };
    }
}
#endif // ! PROJECT_BASE_FUNCTION_SOLUTION_FUNCTION_GRAPH_H