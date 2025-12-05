// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Czech Technical University in Prague

#include <memory>
#include <string>
#include <vector>

#include <depthai/device/Device.hpp>
#include <depthai/pipeline/Pipeline.hpp>
#if DEPTHAI_VERSION_MAJOR < 3
#include <depthai-shared/common/CameraBoardSocket.hpp>
#else
#include <depthai/common/CameraBoardSocket.hpp>
#endif

#include <depthai_ros_driver/dai_nodes/base_node.hpp>
#include <depthai_ros_driver/dai_nodes/nn/nn_helpers.hpp>
#include <depthai_ros_driver/dai_nodes/nn/nn_wrapper.hpp>
#include <depthai_ros_driver/dai_nodes/nn/spatial_nn_wrapper.hpp>
#include <depthai_ros_driver/dai_nodes/sensors/imu.hpp>
#include <depthai_ros_driver/dai_nodes/sensors/sensor_helpers.hpp>
#include <depthai_ros_driver/dai_nodes/sensors/sensor_wrapper.hpp>
#include <depthai_ros_driver/dai_nodes/sensors/stereo.hpp>
#include <depthai_ros_driver/pipeline/base_types.hpp>
#if DEPTHAI_VERSION_MAJOR >= 3
#include <depthai_ros_driver/dai_nodes/sensors/vio.hpp>
#endif
#include <depthai_ros_driver/pipeline/base_pipeline.hpp>
#include <depthai_ros_driver/utils.hpp>
#include <rclcpp/node.hpp>
#include <road_segmentation_plugin/nn_wrapper_road_segmentation.hpp>
#include <road_segmentation_plugin/road_segmentation.hpp>
#include <road_segmentation_plugin/road_segmentation_plugin.hpp>


namespace depthai_ros_driver::pipeline_gen
{

std::unique_ptr<dai_nodes::BaseNode> addNnNode(
  std::shared_ptr<rclcpp::Node> node, std::shared_ptr<dai::Pipeline> pipeline,
#if DEPTHAI_VERSION_MAJOR >= 3
  const std::string& deviceName, const bool rsCompat,
#endif
  dai_nodes::SensorWrapper& sensor)
{
  using dai_nodes::sensor_helpers::NodeNameEnum;
  auto nn = std::make_unique<dai_nodes::NNWrapperRoadSegmentation>(getNodeName(node, NodeNameEnum::NN), node, pipeline,
#if DEPTHAI_VERSION_MAJOR >= 3
  deviceName, rsCompat,
#endif
    sensor);
  return nn;
}

std::vector<std::unique_ptr<dai_nodes::BaseNode>> RGBDRoadSegmentation::createPipeline(
  std::shared_ptr<rclcpp::Node> node, std::shared_ptr<dai::Device> device, std::shared_ptr<dai::Pipeline> pipeline,
#if DEPTHAI_VERSION_MAJOR >= 3
  std::shared_ptr<param_handlers::PipelineGenParamHandler> ph, const std::string& deviceName, bool rsCompat,
#endif
  const std::string& /*nnType*/)
{
  using dai_nodes::sensor_helpers::NodeNameEnum;

  std::vector<std::unique_ptr<dai_nodes::BaseNode>> daiNodes;
#if DEPTHAI_VERSION_MAJOR < 3
  auto rgb = std::make_unique<dai_nodes::SensorWrapper>(
    getNodeName(node, NodeNameEnum::RGB), node, pipeline, device, dai::CameraBoardSocket::CAM_A);
  auto stereo = std::make_unique<dai_nodes::Stereo>(getNodeName(node, NodeNameEnum::Stereo), node, pipeline, device);
  auto nn = addNnNode(node, pipeline, *rgb);
#else
  auto rgb = std::make_unique<dai_nodes::SensorWrapper>(
    getNodeName(node, NodeNameEnum::RGB), node, pipeline, deviceName, rsCompat, dai::CameraBoardSocket::CAM_A);
  auto stereo = std::make_unique<dai_nodes::Stereo>(
    getNodeName(node, NodeNameEnum::Stereo), node, pipeline, device, rsCompat);
  auto nn = pipeline_gen::addNnNode(node, pipeline, deviceName, rsCompat, *rgb);

  if (stereo->isAligned() && stereo->getSocketID() == rgb->getSocketID())
  {
    rgb->getDefaultOut()->link(stereo->getInput(static_cast<int>(dai_nodes::link_types::StereoLinkType::align)));
  }

  addRgbdNode(daiNodes, node, device, pipeline, ph, rsCompat, *rgb, *stereo);

  if (checkForImu(ph, device, node->get_logger()))
  {
    auto imu = std::make_unique<dai_nodes::Imu>("imu", node, pipeline, device, rsCompat);
    if (ph->getParam<bool>("i_enable_vio"))
    {
      auto vio = std::make_unique<dai_nodes::Vio>("vio", node, pipeline, device, rsCompat, *stereo, *imu);
      daiNodes.push_back(std::move(vio));
    }
    daiNodes.push_back(std::move(imu));
  }
#endif

  daiNodes.push_back(std::move(nn));
  daiNodes.push_back(std::move(rgb));
  daiNodes.push_back(std::move(stereo));

  return daiNodes;
}
}

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  depthai_ros_driver::pipeline_gen::RGBDRoadSegmentation,
  depthai_ros_driver::pipeline_gen::BasePipeline)
