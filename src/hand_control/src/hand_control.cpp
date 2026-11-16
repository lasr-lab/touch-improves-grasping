#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lasr_msgs/msg/robot_hand.hpp"
#include "HandModel.hpp"

class HandControlNode : public rclcpp::Node
{
public:
    HandControlNode()
        : Node("hand_control")
    {
        rightHandPublisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/th_right/joint_targets", 1);
        leftHandPublisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/th_left/joint_targets", 1);
        fingerVisPublisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/visualization/hand", 1);

        robotHandSub_ = this->create_subscription<lasr_msgs::msg::RobotHand>(
            "/controls/hand/request", 1,
            std::bind(&HandControlNode::callbackHandRequest, this, std::placeholders::_1));

        Finger zero;
        Thumb tzero;
        Hand currentHand(tzero, zero, zero, zero);
        model_ = std::make_unique<HandModel>(std::move(currentHand));
    }

private:
    void callbackHandRequest(const lasr_msgs::msg::RobotHand::SharedPtr msg)
    {
        Hand newHand(*msg);

        std::vector<visualization_msgs::msg::Marker> markerVector;
        if (!model_->update(std::move(newHand)))
        {
            model_->visMessageFromModel(&markerVector);
            visualizeHand(markerVector);
            return;
        }
        model_->visMessageFromModel(&markerVector);
        visualizeHand(markerVector);

        auto handOut = std_msgs::msg::Float64MultiArray();
        model_->handMessageFromModel(handOut);
        if (msg->left_hand)
            leftHandPublisher_->publish(handOut);
        else
            rightHandPublisher_->publish(handOut);
    }

    void visualizeHand(const std::vector<visualization_msgs::msg::Marker>& markerVector)
    {
        auto out = visualization_msgs::msg::MarkerArray();
        out.markers = markerVector;
        fingerVisPublisher_->publish(out);
    }

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr rightHandPublisher_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr leftHandPublisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr fingerVisPublisher_;
    rclcpp::Subscription<lasr_msgs::msg::RobotHand>::SharedPtr robotHandSub_;

    std::unique_ptr<HandModel> model_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HandControlNode>());
    rclcpp::shutdown();
    return 0;
}
