#pragma once

// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Czech Technical University in Prague

#include <memory>
#include <string>
#include <vector>

#if DEPTHAI_VERSION_MAJOR < 3
#include <depthai-shared/common/CameraBoardSocket.hpp>
#else
#include <depthai/common/CameraBoardSocket.hpp>
#endif

#include <depthai_ros_driver/dai_nodes/base_node.hpp>
#include <depthai_ros_driver/dai_nodes/sensors/sensor_wrapper.hpp>

namespace dai
{
class Pipeline;
class Device;
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

namespace dai_nodes
{
class NNWrapperRoadSegmentation : public BaseNode
{
public:
  explicit NNWrapperRoadSegmentation(const std::string& daiNodeName, std::shared_ptr<rclcpp::Node> node,
    std::shared_ptr<dai::Pipeline> pipeline,
#if DEPTHAI_VERSION_MAJOR >= 3
    const std::string& deviceName, bool rsCompat,
#endif
    dai_nodes::SensorWrapper& sensor, const dai::CameraBoardSocket& socket = dai::CameraBoardSocket::CAM_A);
  ~NNWrapperRoadSegmentation() override;

  void updateParams(const std::vector<rclcpp::Parameter>& params) override;
  void setupQueues(std::shared_ptr<dai::Device> device) override;
  void closeQueues() override;
  void setNames() override;
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
  std::unique_ptr<param_handlers::NNParamHandler> ph;
  std::unique_ptr<BaseNode> nnNode;
};

}
}
