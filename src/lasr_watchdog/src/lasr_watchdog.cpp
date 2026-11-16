#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "std_msgs/msg/bool.hpp"

using namespace std::chrono_literals;

class WatchdogNode : public rclcpp::Node
{
public:
    explicit WatchdogNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("lasr_watchdog", options)
    {
        timeout_ms_ = this->declare_parameter("timeout_ms", 1000);

        status_publisher_ = this->create_publisher<std_msgs::msg::Bool>("/watchdog/status", 1);

        setupSubscriptions();
        setupTimers();

        publish_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(timeout_ms_), std::bind(&WatchdogNode::publishStatus, this));
    }

private:
    void setupSubscriptions()
    {
        auto create_image_callback
            = [this](size_t idx) { return [this, idx](const sensor_msgs::msg::Image::SharedPtr) { markValid(idx); }; };

        auto create_joint_callback = [this](size_t idx)
        { return [this, idx](const sensor_msgs::msg::JointState::SharedPtr) { markValid(idx); }; };

        digit_subs_[0]
            = this->create_subscription<sensor_msgs::msg::Image>("/digit_rgb/index_0", 1, create_image_callback(0));
        digit_subs_[1]
            = this->create_subscription<sensor_msgs::msg::Image>("/digit_rgb/index_1", 1, create_image_callback(1));
        digit_subs_[2]
            = this->create_subscription<sensor_msgs::msg::Image>("/digit_rgb/index_2", 1, create_image_callback(2));
        digit_subs_[3]
            = this->create_subscription<sensor_msgs::msg::Image>("/digit_rgb/index_3", 1, create_image_callback(3));

        digit360_subs_[0]
            = this->create_subscription<sensor_msgs::msg::Image>("/image_raw/index_0", 1, create_image_callback(0));
        digit360_subs_[1]
            = this->create_subscription<sensor_msgs::msg::Image>("/image_raw/index_1", 1, create_image_callback(1));
        digit360_subs_[2]
            = this->create_subscription<sensor_msgs::msg::Image>("/image_raw/index_2", 1, create_image_callback(2));
        digit360_subs_[3]
            = this->create_subscription<sensor_msgs::msg::Image>("/image_raw/index_3", 1, create_image_callback(3));

        arm_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/xarm/joint_states", 1, create_joint_callback(4));
        hand_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/th_right/joint_states", 1, create_joint_callback(5));
        camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/camera/color/image_raw", 1, create_image_callback(6));
    }

    void setupTimers()
    {
        auto create_timeout_callback
            = [this](size_t idx, const std::string& name) { return [this, idx, name]() { markInvalid(idx, name); }; };

        digit_timers_[0]
            = this->create_wall_timer(std::chrono::milliseconds(timeout_ms_), create_timeout_callback(0, "digit0"));
        digit_timers_[1]
            = this->create_wall_timer(std::chrono::milliseconds(timeout_ms_), create_timeout_callback(1, "digit1"));
        digit_timers_[2]
            = this->create_wall_timer(std::chrono::milliseconds(timeout_ms_), create_timeout_callback(2, "digit2"));
        digit_timers_[3]
            = this->create_wall_timer(std::chrono::milliseconds(timeout_ms_), create_timeout_callback(3, "digit3"));

        arm_timer_ = this->create_wall_timer(std::chrono::milliseconds(timeout_ms_), create_timeout_callback(4, "arm"));
        hand_timer_
            = this->create_wall_timer(std::chrono::milliseconds(timeout_ms_), create_timeout_callback(5, "hand"));
        camera_timer_
            = this->create_wall_timer(std::chrono::milliseconds(timeout_ms_), create_timeout_callback(6, "camera"));
    }

    void markValid(size_t idx)
    {
        if (idx < valid_.size())
        {
            valid_[idx] = true;
            resetTimer(idx);
        }
    }

    void markInvalid(size_t idx, const std::string& name)
    {
        if (idx < valid_.size())
        {
            valid_[idx] = false;
            RCLCPP_WARN(this->get_logger(), "Timeout: %s", name.c_str());
        }
    }

    void resetTimer(size_t idx)
    {
        switch (idx)
        {
        case 0:
            digit_timers_[0]->reset();
            break;
        case 1:
            digit_timers_[1]->reset();
            break;
        case 2:
            digit_timers_[2]->reset();
            break;
        case 3:
            digit_timers_[3]->reset();
            break;
        case 4:
            arm_timer_->reset();
            break;
        case 5:
            hand_timer_->reset();
            break;
        case 6:
            camera_timer_->reset();
            break;
        }
    }

    void publishStatus()
    {
        bool all_valid = true;
        for (size_t i = 0; i < valid_.size(); ++i)
        {
            if (!valid_[i])
            {
                all_valid = false;
                break;
            }
        }

        auto msg = std_msgs::msg::Bool();
        msg.data = all_valid;
        status_publisher_->publish(msg);
    }

    int timeout_ms_;
    std::array<bool, 7> valid_ = { false };

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr status_publisher_;
    rclcpp::TimerBase::SharedPtr publish_timer_;

    std::array<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr, 4> digit_subs_;
    std::array<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr, 4> digit360_subs_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr arm_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr hand_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;

    std::array<rclcpp::TimerBase::SharedPtr, 4> digit_timers_;
    rclcpp::TimerBase::SharedPtr arm_timer_;
    rclcpp::TimerBase::SharedPtr hand_timer_;
    rclcpp::TimerBase::SharedPtr camera_timer_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WatchdogNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}