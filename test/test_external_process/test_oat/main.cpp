#include <oatpp/web/server/HttpRequestHandler.hpp>
#include <oatpp/web/server/HttpConnectionHandler.hpp>
#include <oatpp/network/tcp/server/ConnectionProvider.hpp>
#include <oatpp/network/Server.hpp>

#define O_UNUSED(x) (void)x;

// 自定义请求处理程序
class Handler : public oatpp::web::server::HttpRequestHandler
{
public:
    // 处理传入的请求，并返回响应
    std::shared_ptr<OutgoingResponse> handle(const std::shared_ptr<IncomingRequest>& request) override {
        O_UNUSED(request);

        return ResponseFactory::createResponse(Status::CODE_200, "Hello, World!");
    }
};

void run()
{
    // 为 HTTP 请求创建路由器
    auto router = oatpp::web::server::HttpRouter::createShared();

    // 路由 GET - "/hello" 请求到处理程序
    // 浏览器访问：127.0.0.1:8000/hello
    router->route("GET", "/hello", std::make_shared<Handler>());

    // 创建 HTTP 连接处理程序
    auto connectionHandler = oatpp::web::server::HttpConnectionHandler::createShared(router);

    // 创建 TCP 连接提供者
    auto connectionProvider = oatpp::network::tcp::server::ConnectionProvider::createShared({"localhost", 8000, oatpp::network::Address::IP_4});

    // 创建服务器，它接受提供的 TCP 连接并将其传递给 HTTP 连接处理程序
    oatpp::network::Server server(connectionProvider, connectionHandler);

    // 打印服务器端口
    OATPP_LOGI("MyApp", "Server running on port %s", connectionProvider->getProperty("port").getData());

    // 运行服务器
    server.run();
};

int main()
{
    // 初始化 oatpp 环境
    oatpp::base::Environment::init();

    // 运行应用
    run();

    // 销毁 oatpp 环境
    oatpp::base::Environment::destroy();

    return 0;
}
