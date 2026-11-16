#include <array>
#include <cmath>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "lasr_msgs/msg/finger.hpp"
#include "lasr_msgs/msg/finger_position.hpp"
#include "lasr_msgs/msg/hand_position.hpp"
#include "lasr_msgs/msg/robot_hand.hpp"

#include "Finger.hpp"

#include <Eigen/Dense>

namespace
{
constexpr float THUMB_BASE_ROT_OFFSET = 60.0f;
constexpr float THUMB_BASE_JOINT_OFFSET = 40.0f;
constexpr float FINGER_BASE_ROT_LIMIT = 20.0f;
constexpr int QUEUE_DEPTH = 1;

constexpr auto TOPIC_HAND_REQUEST = "/vr/hand/request";
constexpr auto TOPIC_HAND_CONTROL = "/controls/hand/request";
} // namespace

class VrController : public rclcpp::Node
{
public:
    VrController()
        : Node("vr_controller")
    {
        handSub_ = create_subscription<lasr_msgs::msg::HandPosition>(TOPIC_HAND_REQUEST, QUEUE_DEPTH,
            [this](lasr_msgs::msg::HandPosition::ConstSharedPtr msg) { callbackHandTracking(msg); });

        handRequestPublisher_ = create_publisher<lasr_msgs::msg::RobotHand>(TOPIC_HAND_CONTROL, QUEUE_DEPTH);
    }

private:
    static void fingerPositionToFinger(const lasr_msgs::msg::FingerPosition& msg, Finger* finger)
    {
        finger->firstJointPosition
            = Eigen::Vector3d(msg.first_joint_position.x, msg.first_joint_position.y, msg.first_joint_position.z);
        finger->secondJointPosition
            = Eigen::Vector3d(msg.second_joint_position.x, msg.second_joint_position.y, msg.second_joint_position.z);
        finger->tipPosition = Eigen::Vector3d(msg.tip_position.x, msg.tip_position.y, msg.tip_position.z);
    }

    // Angle (in degrees) between two 3D vectors using the half-angle tangent formula.
    static double convertPositionsToAngle(const Eigen::Vector3d& origin, const Eigen::Vector3d& target)
    {
        double phi = 2.0
            * std::atan((origin * target.norm() - origin.norm() * target).norm()
                / (origin * target.norm() + origin.norm() * target).norm());
        return phi * 180.0 / M_PI;
    }

    static void fingerToAngles(lasr_msgs::msg::Finger* out, const Finger* origin, const Finger* target)
    {
        // Spread angle (X-Z plane)
        Eigen::Vector3d tmpTarget(target->firstJointPosition[0], 0.0, target->firstJointPosition[2]);
        Eigen::Vector3d tmpOrigin(origin->firstJointPosition[0], 0.0, origin->firstJointPosition[2]);
        out->finger_base_rot = convertPositionsToAngle(tmpOrigin, tmpTarget);
        if (tmpOrigin[0] < tmpTarget[0])
        {
            out->finger_base_rot *= -1.0;
        }
        out->finger_base_rot = std::clamp(out->finger_base_rot, -FINGER_BASE_ROT_LIMIT, FINGER_BASE_ROT_LIMIT);

        // Curl angle (first joint)
        out->finger_base_joint = convertPositionsToAngle(origin->firstJointPosition, target->firstJointPosition);

        // Second joint and tip
        out->finger_first_joint = convertPositionsToAngle(origin->secondJointPosition, target->secondJointPosition);
        out->finger_tip = convertPositionsToAngle(origin->tipPosition, target->tipPosition);
    }

    void callbackHandTracking(const lasr_msgs::msg::HandPosition::ConstSharedPtr& msg)
    {
        if (msg->is_origin)
        {
            fingerPositionToFinger(msg->thumb, &thumbOrigin_);
            fingerPositionToFinger(msg->index, &indexOrigin_);
            fingerPositionToFinger(msg->middle, &middleOrigin_);
            fingerPositionToFinger(msg->ring, &ringOrigin_);
            return;
        }

        Finger newThumb, newIndex, newMiddle, newRing;
        fingerPositionToFinger(msg->thumb, &newThumb);
        fingerPositionToFinger(msg->index, &newIndex);
        fingerPositionToFinger(msg->middle, &newMiddle);
        fingerPositionToFinger(msg->ring, &newRing);

        auto out = lasr_msgs::msg::RobotHand();
        fingerToAngles(&out.thumb, &thumbOrigin_, &newThumb);
        out.thumb.finger_base_rot += THUMB_BASE_ROT_OFFSET;
        out.thumb.finger_base_joint += THUMB_BASE_JOINT_OFFSET;

        fingerToAngles(&out.index_finger, &indexOrigin_, &newIndex);
        fingerToAngles(&out.middle_finger, &middleOrigin_, &newMiddle);
        fingerToAngles(&out.ring_finger, &ringOrigin_, &newRing);

        out.left_hand = false;
        handRequestPublisher_->publish(out);
    }

    rclcpp::Subscription<lasr_msgs::msg::HandPosition>::SharedPtr handSub_;
    rclcpp::Publisher<lasr_msgs::msg::RobotHand>::SharedPtr handRequestPublisher_;

    Finger thumbOrigin_;
    Finger indexOrigin_;
    Finger middleOrigin_;
    Finger ringOrigin_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VrController>());
    rclcpp::shutdown();
    return 0;
}
