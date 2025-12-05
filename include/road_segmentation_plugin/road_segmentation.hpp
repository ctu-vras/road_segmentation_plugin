#pragma once

// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Czech Technical University in Prague

#include <memory>
#include <string>
#include <vector>

#if DEPTHAI_VERSION_MAJOR < 3
#include <depthai-shared/common/CameraBoardSocket.hpp>
#include <depthai/device/DataQueue.hpp>
#else
#include <depthai/common/CameraBoardSocket.hpp>
#include <depthai/pipeline/MessageQueue.hpp>
#include <depthai_ros_driver/param_handlers/nn_param_handler.hpp>
#endif

#include <cv_bridge/cv_bridge.hpp>
#include <depthai_ros_driver/dai_nodes/base_node.hpp>
#include <depthai_ros_driver/dai_nodes/sensors/sensor_wrapper.hpp>
#include <image_transport/camera_publisher.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

namespace dai
{
class Pipeline;
class Device;
class DataOutputQueue;
class ADatatype;

namespace node
{
class NeuralNetwork;
class ImageManip;
class XLinkOut;
}

namespace ros
{
class ImageConverter;
}

}  // namespace dai

namespace camera_info_manager
{
class CameraInfoManager;
}

namespace rclcpp
{
class Node;
class Parameter;
}

namespace depthai_ros_driver
{
namespace param_handlers
{
class NNParamHandler;
}

namespace dai_nodes::nn
{

enum class SegmentationClass
{
  BACKGROUND,
  ROAD,
  SKY,
};

#if DEPTHAI_VERSION_MAJOR >= 3
class NNParamHandler : public param_handlers::NNParamHandler
{
public:
  NNParamHandler(std::shared_ptr<rclcpp::Node> node, const std::string& name, const std::string& deviceName,
    bool rsCompat, const dai::CameraBoardSocket& socket = dai::CameraBoardSocket::CAM_A);

  void declareParams(std::shared_ptr<dai::node::NeuralNetwork> nn, std::shared_ptr<dai::node::ImageManip> imageManip);

  size_t width;
  size_t height;

private:
  void setImageManip(const std::string& model_path, std::shared_ptr<dai::node::ImageManip> imageManip);
};
#endif

class RoadSegmentation : public BaseNode
{
public:
  RoadSegmentation(
    const std::string& daiNodeName, std::shared_ptr<rclcpp::Node> node, std::shared_ptr<dai::Pipeline> pipeline,
#if DEPTHAI_VERSION_MAJOR >= 3
    const std::string& deviceName, bool rsCompat,
#endif
    dai_nodes::SensorWrapper& camNode, const dai::CameraBoardSocket& socket = dai::CameraBoardSocket::CAM_A);
  ~RoadSegmentation() override;

  void updateParams(const std::vector<rclcpp::Parameter>& params) override;
  void setupQueues(std::shared_ptr<dai::Device> device) override;
  void setNames() override;
  void closeQueues() override;

#if DEPTHAI_VERSION_MAJOR < 3
  void link(dai::Node::Input in, int linkType) override;
  dai::Node::Input getInput(int linkType) override;
  void setXinXout(std::shared_ptr<dai::Pipeline> pipeline) override;
#else
  void link(dai::Node::Input& in, int linkType) override;
  dai::Node::Input& getInput(int linkType) override;
  void setInOut(std::shared_ptr<dai::Pipeline> pipeline) override;
#endif

private:
  void segmentationCB(const std::string& name, const std::shared_ptr<dai::ADatatype>& data);
  void process_frame(std::vector<float>& nn_output, cv::Mat& mask, cv::Mat& entropy, cv::Mat& cost, int img_width,
                     int img_height);

  cv::Mat count_normalized_entropy(const cv::Mat& segm);

  std::vector<std::string> labelNames;
#if DEPTHAI_VERSION_MAJOR < 3
  std::shared_ptr<dai::ros::ImageConverter> imageConverterPt;
#else
  std::shared_ptr<depthai_bridge::ImageConverter> imageConverterPt;
#endif
  std::shared_ptr<camera_info_manager::CameraInfoManager> infoManager;
  image_transport::CameraPublisher ptPub, nnPub_mask;
  image_transport::Publisher nnPub_entropy, nnPub_cost;
  sensor_msgs::msg::CameraInfo nnInfo;
  std::shared_ptr<dai::node::NeuralNetwork> segNode;
  std::shared_ptr<dai::node::ImageManip> imageManip;
#if DEPTHAI_VERSION_MAJOR < 3
  std::unique_ptr<param_handlers::NNParamHandler> ph;
  std::shared_ptr<dai::DataOutputQueue> nnQ, ptQ;
#else
  std::unique_ptr<NNParamHandler> ph;
  std::shared_ptr<dai::MessageQueue> nnQ, ptQ;
#endif
  std::shared_ptr<dai::node::XLinkOut> xoutNN, xoutPT;
  std::string nnQName, ptQName;
  std::string frame;
  std::array<float, 3> classCosts;
  float normalizedEntropyThreshold{1.0};

  std::chrono::time_point<std::chrono::steady_clock> steadyBaseTime;
  rclcpp::Time rosBaseTime;
  int64_t totalNsChange{0};
};

}
}
