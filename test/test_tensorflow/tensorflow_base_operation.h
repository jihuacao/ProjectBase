#include <tensorflow/core/public/session.h>
#include <glog/logging.h>

void a(){
    const std::string pathToGraph = std::string(LoadRootPath) + "LoadModelByMeta/model_test.meta";
	const std::string checkpointPath = std::string(LoadRootPath) + "LoadModelByMeta/model_test";

	auto session = tf::NewSession(tf::SessionOptions());
	if (session == nullptr)
	{
		throw std::runtime_error("Could not create Tensorflow session.");
	}

	tf::Status status;

	// Read in the protobuf graph we exported
	tf::MetaGraphDef graph_def;
	status = tf::ReadBinaryProto(tf::Env::Default(), pathToGraph, &graph_def);
	DLOG(INFO) << graph_def.DebugString();
	if (!status.ok())
	{
		throw std::runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
	}

	// Add the graph to the session
	status = session->Create(graph_def.graph_def());
	if (!status.ok())
	{
		throw std::runtime_error("Error creating graph: " + status.ToString());
	}

	// Read weights from the saved checkpoint
	tf::Tensor checkpointPathTensor(tf::DT_STRING, tf::TensorShape());
	checkpointPathTensor.scalar<std::string>()() = checkpointPath;

	status = session->Run({ { graph_def.saver_def().filename_tensor_name(), checkpointPathTensor }, }, {},
	{ graph_def.saver_def().restore_op_name() }, nullptr);
	if (!status.ok())
	{
		throw std::runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
	}
	
	std::vector<tf::Tensor> init_output;
	status = session->Run({}, { "var1/read", "var2/read" }, {}, &init_output);
	if (!status.ok())
	{
		throw std::runtime_error("Error creating graph: " + status.ToString());
	}
	for (auto iter = init_output.begin(); iter != init_output.end(); ++iter) {
		DLOG(INFO) << iter->scalar<float>()();
	}
	std::vector<tf::Tensor> output;
	tf::Tensor input = tf::Tensor(tf::DT_FLOAT, {});
	input.scalar<float>()() = 9.0;
	status = session->Run({ {"default/input", input } }, { "default/add" }, {}, &output);
	if (!status.ok())
	{
		throw std::runtime_error("Error creating graph: " + status.ToString());
	}
	for (auto iter = output.begin(); iter != output.end(); ++iter) {
		DLOG(INFO) << iter->scalar<float>()();
	}
}