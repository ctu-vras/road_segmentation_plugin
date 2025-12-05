// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Czech Technical University in Prague

#include <memory>
#include <string>

#include <depthai/device/Device.hpp>
#include <depthai/pipeline/Pipeline.hpp>

#include <depthai_ros_driver/dai_nodes/nn/segmentation.hpp>
#include <depthai_ros_driver/param_handlers/nn_param_handler.hpp>
#include <rclcpp/node.hpp>
#include <road_segmentation_plugin/nn_wrapper_road_segmentation.hpp>
#include <road_segmentation_plugin/road_segmentation.hpp>

namespace depthai_ros_driver::dai_nodes
{

NNWrapperRoadSegmentation::NNWrapperRoadSegmentation(
  const std::string& daiNodeName, std::shared_ptr<rclcpp::Node> node, std::shared_ptr<dai::Pipeline> pipeline,
#if DEPTHAI_VERSION_MAJOR >= 3
  const std::string& deviceName, const bool rsCompat,
#endif
  dai_nodes::SensorWrapper& camNode, const dai::CameraBoardSocket& socket)
  : BaseNode(daiNodeName, node, pipeline
#if DEPTHAI_VERSION_MAJOR >= 3
      , deviceName, rsCompat
#endif
  )
{
  RCLCPP_DEBUG(node->get_logger(), "Creating node %s base", daiNodeName.c_str());

  ph = std::make_unique<param_handlers::NNParamHandler>(node, daiNodeName,
#if DEPTHAI_VERSION_MAJOR >= 3
    deviceName, rsCompat,
#endif
    socket);

  nnNode = std::make_unique<dai_nodes::nn::RoadSegmentation>(getName(), getROSNode(), pipeline,
#if DEPTHAI_VERSION_MAJOR >= 3
    deviceName, rsCompat,
#endif
    camNode, socket);

  RCLCPP_DEBUG(node->get_logger(), "Base node %s created", daiNodeName.c_str());
}

NNWrapperRoadSegmentation::~NNWrapperRoadSegmentation() = default;

void NNWrapperRoadSegmentation::setNames()
{
}

void NNWrapperRoadSegmentation::setupQueues(std::shared_ptr<dai::Device> device)
{
  nnNode->setupQueues(device);
}

void NNWrapperRoadSegmentation::closeQueues()
{
  nnNode->closeQueues();
}

#if DEPTHAI_VERSION_MAJOR < 3
void NNWrapperRoadSegmentation::setXinXout(std::shared_ptr<dai::Pipeline> /*pipeline*/)
{
}

void NNWrapperRoadSegmentation::link(const dai::Node::Input in, const int linkType)
{
  nnNode->link(in, linkType);
}

dai::Node::Input NNWrapperRoadSegmentation::getInput(const int linkType)
{
  return nnNode->getInput(linkType);
}
#else
void NNWrapperRoadSegmentation::setInOut(std::shared_ptr<dai::Pipeline> /*pipeline*/)
{
}

void NNWrapperRoadSegmentation::link(dai::Node::Input& in, const int linkType)
{
  nnNode->link(in, linkType);
}

dai::Node::Input& NNWrapperRoadSegmentation::getInput(const int linkType)
{
  return nnNode->getInput(linkType);
}
#endif

void NNWrapperRoadSegmentation::updateParams(const std::vector<rclcpp::Parameter>& params)
{
  // ph->setRuntimeParams(params);
  nnNode->updateParams(params);
}

}
