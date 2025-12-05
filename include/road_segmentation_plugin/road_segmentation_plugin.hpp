#pragma once

// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Czech Technical University in Prague

#include <memory>
#include <string>
#include <vector>

#include <depthai_ros_driver/pipeline/base_pipeline.hpp>

namespace dai
{
class Pipeline;
class Device;
}

namespace dai_nodes
{
class BaseNode;
}

namespace rclcpp
{
class Node;
}

namespace depthai_ros_driver::pipeline_gen
{

class RGBDRoadSegmentation : public BasePipeline
{
public:
  std::vector<std::unique_ptr<dai_nodes::BaseNode>> createPipeline(
    std::shared_ptr<rclcpp::Node> node, std::shared_ptr<dai::Device> device, std::shared_ptr<dai::Pipeline> pipeline,
#if DEPTHAI_VERSION_MAJOR >= 3
    std::shared_ptr<param_handlers::PipelineGenParamHandler> ph, const std::string& deviceName, bool rsCompat,
#endif
    const std::string& nnType) override;
};

}
