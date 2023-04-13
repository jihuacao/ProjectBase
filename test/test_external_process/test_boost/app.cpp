#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/beast/core/detail/base64.hpp>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <boost/beast.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core/detail/base64.hpp>

#include <boost/json/src.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

typedef struct {
    boost::string_view taskid;
    boost::string_view msg;
} ErrorPack;

// Return a response for the given request.
//
// The concrete type of the response message (which depends on the
// request), is type-erased in message_generator.
template <class Body, class Allocator>
http::message_generator
handle_request(http::request<Body, http::basic_fields<Allocator>>&& req)
{
    // Returns a bad request response
    auto const failed_call =
    [&req](const ErrorPack& ep)
    {
        http::response<http::string_body> res{http::status::bad_request, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "application/json");
        res.keep_alive(req.keep_alive());
        boost::json::object data;
        data["taskID"] = ep.taskid;
        data["isSuccess"] = false;
        data["message"] = ep.msg;
        res.body() = boost::json::serialize(data);
        res.prepare_payload();
        return res;
    };

    // Make sure we can handle the method
    if( req.method() != http::verb::get && req.method() != http::verb::post){
        auto ep = ErrorPack();
        ep.taskid = "";
        ep.msg = "Unknown HTTP-method";
        return failed_call(ep);
    }

    // Request path must be absolute and not contain "..".
    if( req.target().empty() || req.target()[0] != '/' || req.target().compare(boost::string_view("/api"))){
        auto ep = ErrorPack();
        ep.taskid = "";
        ep.msg = "Illegal request-target";
        return failed_call(ep);
    }
    
    boost::json::error_code json_ec;
    auto reqj = boost::json::value_to<boost::json::object>(boost::json::parse(req.body(), json_ec));
    auto taskidi = reqj.find("taskID");
    boost::string_view taskid("");
    if (taskidi == reqj.end()){
        auto ep = ErrorPack();
        ep.taskid = "";
        ep.msg = "taskID not found";
        return failed_call(ep);
    }
    else{
        taskid = boost::json::value_to<boost::string_view>(taskidi->value());
    }

    auto imgbase64i = reqj.find("imgBase64");
    cv::Mat img;
    if (imgbase64i == reqj.end()){
        auto ep = ErrorPack();
        ep.taskid = taskid;
        ep.msg = "imgBase64 not found";
        return failed_call(ep);
    }
    else{
        boost::string_view imagebase64 = boost::json::value_to<boost::string_view>(imgbase64i->value());
        // todo: decode base64 and change to opencv
        std::size_t len = imagebase64.size();
        std::string output;
        output.resize(boost::beast::detail::base64::decoded_size(len));
        auto result = boost::beast::detail::base64::decode(&output[0], imagebase64.data(), len);
        output.resize(result.first);
        // todo: string to opencv image
        std::vector<unsigned char> data(output.begin(), output.end());
        img = cv::imdecode(data, cv::IMREAD_UNCHANGED);
        //cv::imwrite("a.jpg", img);
    }

    // todo: do the predict

    // Attempt to open the file
    http::string_body::value_type body;
    boost::json::object data;
    data["taskID"] = "task";
    data["isSuccess"] = true;
    data["message"] = "OK";
    boost::json::object result;
    result["isAlarm"] = false;
    result["timestamp"] = "";
    result["timecost"] = "0ms";
    boost::json::array positions;
    int num = 2;
    for (auto n = 0; n < num; ++n){
        positions.push_back(
            {0.0, 0.0, 0.0, 0.0, "0.9", "test"}
        );
    }
    result["positions"] = positions;
    data["result"] = result;
    body = boost::json::serialize(data);

    auto const size = body.size();

    http::response<http::string_body> res{
        std::piecewise_construct,
        std::make_tuple(std::move(body)),
        std::make_tuple(http::status::ok, req.version())};
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "application/json");
    res.content_length(size);
    res.keep_alive(req.keep_alive());
    return res;
}

//------------------------------------------------------------------------------

// Report a failure
void
fail(beast::error_code ec, char const* what)
{
    std::cerr << what << ": " << ec.message() << "\n";
}

// Handles an HTTP server connection
class session : public std::enable_shared_from_this<session>
{
    beast::tcp_stream stream_;
    beast::flat_buffer buffer_;
    //std::shared_ptr<std::string const> doc_root_;
    //http::request<http::string_body> req_;
    http::request_parser<http::string_body> reqp_;

public:
    // Take ownership of the stream
    session(
        tcp::socket&& socket
        //, std::shared_ptr<std::string const> const& doc_root
        )
        : stream_(std::move(socket))
        //, doc_root_(doc_root)
    {
        reqp_.body_limit((std::numeric_limits<std::uint64_t>::max)());
    }

    // Start the asynchronous operation
    void
    run()
    {
        // We need to be executing within a strand to perform async operations
        // on the I/O objects in this session. Although not strictly necessary
        // for single-threaded contexts, this example code is written to be
        // thread-safe by default.
        net::dispatch(stream_.get_executor(),
                      beast::bind_front_handler(
                          &session::do_read,
                          shared_from_this()));
    }

    void
    do_read()
    {
        stream_.expires_after(std::chrono::seconds(30));

        // Read a request
        http::async_read(stream_, buffer_, reqp_,
            beast::bind_front_handler(
                &session::on_read,
                shared_from_this()));
    }

    void
    on_read(
        beast::error_code ec,
        std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        // This means they closed the connection
        if(ec == http::error::end_of_stream)
            return do_close();

        if(ec)
            return fail(ec, "read");

        // Send the response
        send_response(
            handle_request(
                //*doc_root_, 
                std::move(reqp_.get())));
    }

    void
    send_response(http::message_generator&& msg)
    {
        bool keep_alive = msg.keep_alive();

        // Write the response
        beast::async_write(
            stream_,
            std::move(msg),
            beast::bind_front_handler(
                &session::on_write, shared_from_this(), keep_alive));
    }

    void
    on_write(
        bool keep_alive,
        beast::error_code ec,
        std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        if(ec)
            return fail(ec, "write");

        if(! keep_alive)
        {
            // This means we should close the connection, usually because
            // the response indicated the "Connection: close" semantic.
            return do_close();
        }

        // Read another request
        do_read();
    }

    void
    do_close()
    {
        // Send a TCP shutdown
        beast::error_code ec;
        stream_.socket().shutdown(tcp::socket::shutdown_send, ec);

        // At this point the connection is closed gracefully
    }
};

//------------------------------------------------------------------------------

// Accepts incoming connections and launches the sessions
class listener : public std::enable_shared_from_this<listener>
{
    net::io_context& ioc_;
    tcp::acceptor acceptor_;
    //std::shared_ptr<std::string const> doc_root_;

public:
    listener(
        net::io_context& ioc
        , tcp::endpoint endpoint
        //, std::shared_ptr<std::string const> const& doc_root
        )
        : ioc_(ioc)
        , acceptor_(net::make_strand(ioc))
        //, doc_root_(doc_root)
    {
        beast::error_code ec;

        // Open the acceptor
        acceptor_.open(endpoint.protocol(), ec);
        if(ec)
        {
            fail(ec, "open");
            return;
        }

        // Allow address reuse
        acceptor_.set_option(net::socket_base::reuse_address(true), ec);
        if(ec)
        {
            fail(ec, "set_option");
            return;
        }

        // Bind to the server address
        acceptor_.bind(endpoint, ec);
        if(ec)
        {
            fail(ec, "bind");
            return;
        }

        // Start listening for connections
        acceptor_.listen(
            net::socket_base::max_listen_connections, ec);
        if(ec)
        {
            fail(ec, "listen");
            return;
        }
    }

    // Start accepting incoming connections
    void
    run()
    {
        do_accept();
    }

private:
    void
    do_accept()
    {
        // The new connection gets its own strand
        acceptor_.async_accept(
            net::make_strand(ioc_),
            beast::bind_front_handler(
                &listener::on_accept,
                shared_from_this()));
    }

    void
    on_accept(beast::error_code ec, tcp::socket socket)
    {
        if(ec)
        {
            fail(ec, "accept");
            return; // To avoid infinite loop
        }
        else
        {
            // Create the session and run it
            std::make_shared<session>(
                std::move(socket)
                //, doc_root_
                )->run();
        }

        // Accept another connection
        do_accept();
    }
};

//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    // Check command line arguments.
    if (argc != 4)
    {
        std::cerr <<
            "Usage: http-server-async <address> <port> <doc_root> <threads>\n" <<
            "Example:\n" <<
            "    http-server-async 0.0.0.0 8080 . 1\n";
        return EXIT_FAILURE;
    }
    auto const address = net::ip::make_address(argv[1]);
    auto const port = static_cast<unsigned short>(std::atoi(argv[2]));
    //auto const doc_root = std::make_shared<std::string>(argv[3]);
    auto const threads = std::max<int>(1, std::atoi(argv[3]));

    // The io_context is required for all I/O
    net::io_context ioc{threads};

    // Create and launch a listening port
    std::make_shared<listener>(
        ioc,
        tcp::endpoint{address, port}
        //, doc_root
        )->run();

    // Run the I/O service on the requested number of threads
    std::vector<std::thread> v;
    v.reserve(threads - 1);
    for(auto i = threads - 1; i > 0; --i)
        v.emplace_back(
        [&ioc]
        {
            ioc.run();
        });
    ioc.run();

    return EXIT_SUCCESS;
}