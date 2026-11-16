#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <string>
#include <vector>

#include "tilburg_hand_interface/motor_controller.hpp"

void printUsage();

class TilburgHandInterface : public rclcpp::Node
{
public:
    void loadParams()
    {
        std::string usb_port;
        this->declare_parameter("side", rclcpp::PARAMETER_STRING);
        this->declare_parameter("usb-port", rclcpp::PARAMETER_STRING);
        this->declare_parameter("joints-post-rate", rclcpp::PARAMETER_INTEGER);

        this->side_ = this->get_parameter("side").as_string();
        usb_port = this->get_parameter("usb-port").as_string();
        joints_post_rate = this->get_parameter("joints-post-rate").as_int();
        motor_controller_ = new tilburg_hand_interface::MotorController(usb_port, this->side_);

        // Validate arguments
        if (usb_port.empty())
        {
            std::cerr << "Error: --usb-port is required" << std::endl;
            printUsage();
            return;
        }

        if (this->side_.empty())
        {
            std::cerr << "Error: --side is required" << std::endl;
            printUsage();
            return;
        }

        if (this->side_ != "left" && this->side_ != "right")
        {
            std::cerr << "Error: --side must be 'left' or 'right'" << std::endl;
            printUsage();
            return;
        }

        if (this->joints_post_rate <= 0.0)
        {
            std::cerr << "Error: --joints-post-rate must be positive" << std::endl;
            printUsage();
            return;
        }
    }

    TilburgHandInterface()
        : Node("tilburg_hand_interface")
        , joint_names_(16)
    {

        this->loadParams();
        // Initialize joint names
        for (size_t i = 0; i < 16; ++i)
        {
            joint_names_[i] = "joint_" + std::to_string(i);
        }

        // Initialize motor controller
        if (!motor_controller_->initialize())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize motor controller");
            rclcpp::shutdown();
            return;
        }

        // Enable torque and LED
        if (!motor_controller_->enableTorqueAndLED())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to enable torque and LED");
            rclcpp::shutdown();
            return;
        }

        // Create subscriber for joint targets
        std::string target_topic = "/th_" + side_ + "/joint_targets";
        joint_targets_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            target_topic, 10, std::bind(&TilburgHandInterface::jointTargetsCallback, this, std::placeholders::_1));

        // Create publisher for joint states
        std::string state_topic = "/th_" + side_ + "/joint_states";
        joint_states_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(state_topic, 10);

        // Create timer for publishing joint states
        auto period = std::chrono::duration<double>(1.0 / this->joints_post_rate);
        timer_ = this->create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&TilburgHandInterface::publishJointStates, this));

        RCLCPP_INFO(this->get_logger(), "Tilburg Hand Interface initialized for %s hand", side_.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing joint states at %.1f Hz", this->joints_post_rate);
        RCLCPP_INFO(this->get_logger(), "Listening for joint targets on: %s", target_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing joint states on: %s", state_topic.c_str());
    }

    ~TilburgHandInterface() { motor_controller_->shutdown(); }

private:
    void jointTargetsCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() != 16)
        {
            RCLCPP_ERROR(this->get_logger(), "Expected 16 joint positions, received %zu", msg->data.size());
            return;
        }

        std::vector<double> joint_positions(msg->data.begin(), msg->data.end());

        if (!motor_controller_->setJointPositions(joint_positions))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to set joint positions");
        }
    }

    void publishJointStates()
    {
        std::vector<double> positions, velocities, currents;

        if (!motor_controller_->readJointStates(positions, velocities, currents))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to read joint states");
            return;
        }

        auto joint_state_msg = sensor_msgs::msg::JointState();
        joint_state_msg.header.stamp = this->get_clock()->now();
        joint_state_msg.header.frame_id = "";

        joint_state_msg.name = joint_names_;
        joint_state_msg.position = positions;
        joint_state_msg.velocity = velocities;
        joint_state_msg.effort = currents; // Using effort field for current

        joint_states_pub_->publish(joint_state_msg);
    }

    tilburg_hand_interface::MotorController* motor_controller_;
    std::string side_;
    std::vector<std::string> joint_names_;
    double joints_post_rate;

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_targets_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

void printUsage()
{
    std::cout << "Tilburg Hand Interface - ROS2 Dynamixel Motor Controller" << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  ros2 run tilburg_hand_interface tilburg_hand_interface [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << "Required Parameters:" << std::endl;
    std::cout << "  --usb-port <device>     USB port (e.g., /dev/ttyUSB0, /dev/ttyACM0)" << std::endl;
    std::cout << "  --side <left|right>     Hand side" << std::endl;
    std::cout << std::endl;
    std::cout << "Optional Parameters:" << std::endl;
    std::cout << "  --joints-post-rate <hz> Joint state publishing rate in Hz (default: 50)" << std::endl;
    std::cout << "  --help, -h              Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  # Right hand on /dev/ttyUSB0 at 50Hz" << std::endl;
    std::cout << "  ros2 run tilburg_hand_interface tilburg_hand_interface \\" << std::endl;
    std::cout << "    --usb-port /dev/ttyUSB0 --side right --joints-post-rate 50" << std::endl;
    std::cout << std::endl;
    std::cout << "  # Left hand on /dev/ttyACM0 at 100Hz" << std::endl;
    std::cout << "  ros2 run tilburg_hand_interface tilburg_hand_interface \\" << std::endl;
    std::cout << "    --usb-port /dev/ttyACM0 --side left --joints-post-rate 100" << std::endl;
    std::cout << std::endl;
    std::cout << "Topics:" << std::endl;
    std::cout << "  Subscriber: /th_{side}/joint_targets (std_msgs/Float64MultiArray)" << std::endl;
    std::cout << "  Publisher:  /th_{side}/joint_states (sensor_msgs/JointState)" << std::endl;
    std::cout << std::endl;
    std::cout << "Test the interface:" << std::endl;
    std::cout << "  ros2 run tilburg_hand_interface test_hand_control --side <left|right>" << std::endl;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    try
    {
        auto node = std::make_shared<TilburgHandInterface>();
        rclcpp::spin(node);
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("tilburg_hand_interface"), "Exception: %s", e.what());
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
