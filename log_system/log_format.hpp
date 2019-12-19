#ifndef PROJECT_BASE_LOG_SYSTEM_LOG_FORMAT
#define PROJECT_BASE_LOG_SYSTEM_LOG_FORMAT
#include <glog/logging.h>
#include <thread>
#define FunctionInter(FunctionName) DLOG(INFO) << "-->thread: " << std::this_thread::get_id() << " | " << "function: " << #FunctionName
#define FunctionOuter(FunctionName) DLOG(INFO) << "<--thread: " << std::this_thread::get_id() << " | " << "function: " << #FunctionName

#define ApiInter(FunctionName) LOG(INFO) << "-->thread: " << std::this_thread::get_id() << " | " << "function: " << #FunctionName
#define ApiOuter(FunctionName) LOG(INFO) << "<--thread: " << std::this_thread::get_id() << " | " << "function: " << #FunctionName

#define UNIMPLMENT(FunctionName) LOG(FATAL) << "thread: " << std::this_thread::get_id() << " | " << "function: " << #FunctionName

#ifdef _DEBUG
#define DEBUG_TRY_CATCH(command, exception, info) \
try{ \
	command; \
} \
catch (exception ex) { \
	DLOG(FATAL) << "DEBUG_TRY_CATCH: " << info; \
}
#else
#define DEBUG_TRY_CATCH(command, exception, info) \
	command;
#endif

#define THREAD_INFO std::this_thread::get_id()
#else
#endif // ! PROJECT_BASE_LOG_SYSTEM_LOG_FORMAT