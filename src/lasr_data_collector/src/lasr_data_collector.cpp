#include <array>
#include <cstdint>
#include <functional>
#include <rclcpp/rclcpp.hpp>

#include "lasr_msgs/msg/action.hpp"
#include "lasr_msgs/msg/audio.hpp"
#include "lasr_msgs/msg/contact_classification.hpp"
#include "lasr_msgs/msg/qpos.hpp"
#include "lasr_msgs/msg/qvel.hpp"
#include "lasr_msgs/msg/robot_hand_ee.hpp"
#include "lasr_msgs/msg/x_arm_command.hpp"

#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#include "geometry_msgs/msg/vector3.hpp"

#include "d360_msgs/msg/audio_data_d360.hpp"

#include "std_msgs/msg/float64_multi_array.hpp"

#define PUBLISH_FREQUENCY 30 // ms, not actually frequency
#define NUM_SENSORS 4

// --- Validity bitmask ---
// Each sensor/data source gets a named bit. Callbacks set bits in valid_flags;
// publish() checks that all required_flags bits are set.
// To skip a sensor group, clear its bits in required_flags (see main()).
enum SensorBits : uint32_t
{
    S_HAND_COMMAND = (1u << 0),
    S_ARM_COMMAND = (1u << 1),
    S_REALSENSE_IMG = (1u << 2),
    S_HAND_STATE = (1u << 3),
    S_ARM_STATE = (1u << 4),
    S_REALSENSE_DEPTH = (1u << 5),
    S_DIGIT360_0 = (1u << 6),
    S_DIGIT360_1 = (1u << 7),
    S_DIGIT360_2 = (1u << 8),
    S_DIGIT360_3 = (1u << 9),
    S_CLASSIF_0 = (1u << 10),
    S_CLASSIF_1 = (1u << 11),
    S_CLASSIF_2 = (1u << 12),
    S_CLASSIF_3 = (1u << 13),
    S_AUDIO_0 = (1u << 14),
    S_AUDIO_1 = (1u << 15),
    S_AUDIO_2 = (1u << 16),
    S_AUDIO_3 = (1u << 17),
};

static constexpr uint32_t MASK_DIGIT360_ALL = S_DIGIT360_0 | S_DIGIT360_1 | S_DIGIT360_2 | S_DIGIT360_3;
static constexpr uint32_t MASK_CLASSIF_ALL = S_CLASSIF_0 | S_CLASSIF_1 | S_CLASSIF_2 | S_CLASSIF_3;
static constexpr uint32_t MASK_AUDIO_ALL = S_AUDIO_0 | S_AUDIO_1 | S_AUDIO_2 | S_AUDIO_3;
static constexpr uint32_t MASK_ALL = 0x0003'FFFF; // bits 0..17

static const uint32_t S_DIGIT360[NUM_SENSORS] = { S_DIGIT360_0, S_DIGIT360_1, S_DIGIT360_2, S_DIGIT360_3 };
static const uint32_t S_CLASSIF[NUM_SENSORS] = { S_CLASSIF_0, S_CLASSIF_1, S_CLASSIF_2, S_CLASSIF_3 };
static const uint32_t S_AUDIO[NUM_SENSORS] = { S_AUDIO_0, S_AUDIO_1, S_AUDIO_2, S_AUDIO_3 };

uint32_t valid_flags = 0;
uint32_t required_flags = MASK_ALL;

rclcpp::Node::SharedPtr node;

// --- Publishers ---
rclcpp::Publisher<lasr_msgs::msg::Qpos>::SharedPtr qposPublisher;
rclcpp::Publisher<lasr_msgs::msg::Qvel>::SharedPtr qvelPublisher;
rclcpp::Publisher<lasr_msgs::msg::Action>::SharedPtr actionPublisher;
rclcpp::Publisher<lasr_msgs::msg::Action>::SharedPtr actionPublisherEE;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr realsenseImagePublisher;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr realsenseDepthImagePublisher;

std::array<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr, NUM_SENSORS> digit360_publishers;
std::array<rclcpp::Publisher<lasr_msgs::msg::ContactClassification>::SharedPtr, NUM_SENSORS>
    classification_publishers;
std::array<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr, NUM_SENSORS> digitRGB_publishers;
std::array<rclcpp::Publisher<lasr_msgs::msg::Audio>::SharedPtr, NUM_SENSORS> audio_publishers;

rclcpp::TimerBase::SharedPtr publishTimer;

// --- Message buffers ---
rclcpp::Time publishTime;
auto qPosOut = lasr_msgs::msg::Qpos();
auto qVelOut = lasr_msgs::msg::Qvel();
auto actionOut = lasr_msgs::msg::Action();
auto actionOutEE = lasr_msgs::msg::Action();
auto realsenseImageOut = sensor_msgs::msg::Image();
auto realsenseDepthImageOut = sensor_msgs::msg::Image();

std::array<sensor_msgs::msg::Image, NUM_SENSORS> digit360_out;
std::array<lasr_msgs::msg::ContactClassification, NUM_SENSORS> classification_out;
std::array<sensor_msgs::msg::Image, NUM_SENSORS> digitRGB_out;
std::array<lasr_msgs::msg::Audio, NUM_SENSORS> audioOut;

// --- Callbacks ---

void callbackHandCommand(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    actionOut.hand_command = msg->data;
    valid_flags |= S_HAND_COMMAND;
}

void callbackArmCommand(const lasr_msgs::msg::XArmCommand::SharedPtr msg)
{
    actionOut.xarm_command = *msg;
    actionOutEE.xarm_command = *msg;
    valid_flags |= S_ARM_COMMAND;
}

void callbackRealsenseImage(const sensor_msgs::msg::Image::SharedPtr msg)
{
    realsenseImageOut = *msg;
    valid_flags |= S_REALSENSE_IMG;
}

void callbackHandState(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    qPosOut.hand_position = msg->position;
    qVelOut.hand_velocity = msg->velocity;
    valid_flags |= S_HAND_STATE;
}

void callbackArmState(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    qPosOut.arm_position = msg->position;
    qVelOut.arm_velocity = msg->velocity;
    valid_flags |= S_ARM_STATE;
}

void callbackRealsenseDepthImage(const sensor_msgs::msg::Image::SharedPtr msg)
{
    realsenseDepthImageOut = *msg;
    valid_flags |= S_REALSENSE_DEPTH;
}

void callbackHandCommandEE(const lasr_msgs::msg::RobotHandEE::SharedPtr msg)
{
    std::vector<double> ee;
    auto pushVec3 = [&ee](geometry_msgs::msg::Vector3 v)
    {
        ee.push_back(v.x);
        ee.push_back(v.y);
        ee.push_back(v.z);
    };
    pushVec3(msg->thumb);
    pushVec3(msg->index_finger);
    pushVec3(msg->middle_finger);
    pushVec3(msg->ring_finger);
    actionOutEE.hand_command = ee;
}

// --- Publish helpers ---

void publishQpos()
{
    qPosOut.header.stamp = publishTime;
    qposPublisher->publish(qPosOut);
}

void publishQvel()
{
    qVelOut.header.stamp = publishTime;
    qvelPublisher->publish(qVelOut);
}

void publishAction()
{
    actionOut.header.stamp = publishTime;
    actionPublisher->publish(actionOut);
}

void publishActionEE()
{
    actionOutEE.header.stamp = publishTime;
    actionPublisherEE->publish(actionOutEE);
}

void publishImages()
{
    realsenseDepthImageOut.header.stamp = publishTime;
    realsenseDepthImagePublisher->publish(realsenseDepthImageOut);

    realsenseImageOut.header.stamp = publishTime;
    realsenseImagePublisher->publish(realsenseImageOut);

    for (int i = 0; i < NUM_SENSORS; i++)
    {
        digit360_out[i].header.stamp = publishTime;
        digit360_publishers[i]->publish(digit360_out[i]);
    }

    // for (int i = 0; i < NUM_SENSORS; i++)
    // {
    //     digitRGB_out[i].header.stamp = publishTime;
    //     digitRGB_publishers[i]->publish(digitRGB_out[i]);
    // }
}

void publishAudio()
{
    for (int i = 0; i < NUM_SENSORS; i++)
    {
        audioOut[i].header.stamp = publishTime;
        audio_publishers[i]->publish(audioOut[i]);
    }
}

void publishClassification()
{
    for (int i = 0; i < NUM_SENSORS; i++)
    {
        classification_out[i].header.stamp = publishTime;
        classification_publishers[i]->publish(classification_out[i]);
    }
}

void publish()
{
    uint32_t missing = required_flags & ~valid_flags;
    if (missing)
    {
        for (int i = 0; i < 18; i++)
        {
            if (missing & (1u << i))
                std::cout << i << std::endl;
        }
        return;
    }
    publishTime = node->get_clock()->now();
    publishAction();
    publishActionEE();
    publishQpos();
    publishQvel();
    publishImages();
    publishAudio();
    publishClassification();
    publishTimer->reset();
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    node = std::make_shared<rclcpp::Node>("lasr_data_collector");

    // --- Configure required sensors ---
    // Clear bits to skip sensors you don't need. Examples:
    //   required_flags &= ~MASK_AUDIO_ALL;      // skip all audio
    //   required_flags &= ~MASK_CLASSIF_ALL;     // skip all classification
    //   required_flags &= ~MASK_DIGIT360_ALL;    // skip all digit360 images
    //   required_flags &= ~S_DIGIT360_2;         // skip a single sensor
    //   required_flags = MASK_ALL & ~MASK_AUDIO_ALL & ~MASK_CLASSIF_ALL; // combined

    required_flags &= ~MASK_AUDIO_ALL;
    required_flags &= ~MASK_CLASSIF_ALL;

    // --- Subscriptions ---
    auto handCommandSub = node->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/th_right/joint_targets", 1, callbackHandCommand);
    auto handEECommandSub
        = node->create_subscription<lasr_msgs::msg::RobotHandEE>("/ik/hand/request", 1, callbackHandCommandEE);
    auto armCommandSub
        = node->create_subscription<lasr_msgs::msg::XArmCommand>("/xarm/command", 1, callbackArmCommand);
    auto handStateSub
        = node->create_subscription<sensor_msgs::msg::JointState>("/th_right/joint_states", 1, callbackHandState);
    auto armStateSub
        = node->create_subscription<sensor_msgs::msg::JointState>("/xarm/joint_states", 1, callbackArmState);
    auto realsenseImageSub = node->create_subscription<sensor_msgs::msg::Image>(
        "/camera/camera/color/image_raw", 1, callbackRealsenseImage);
    auto realsenseDepthImageSub = node->create_subscription<sensor_msgs::msg::Image>(
        "/camera/camera/depth/image_rect_raw", 1, callbackRealsenseDepthImage);

    // Indexed digit360 image subscriptions
    std::array<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr, NUM_SENSORS> digit360_subs;
    for (int i = 0; i < NUM_SENSORS; i++)
    {
        digit360_subs[i]
            = node->create_subscription<sensor_msgs::msg::Image>("/image_raw/index_" + std::to_string(i), 1,
                [i](const sensor_msgs::msg::Image::SharedPtr msg)
                {
                    digit360_out[i] = *msg;
                    valid_flags |= S_DIGIT360[i];
                });
    }

    // Indexed classification subscriptions
    std::array<rclcpp::Subscription<lasr_msgs::msg::ContactClassification>::SharedPtr, NUM_SENSORS>
        classification_subs;
    for (int i = 0; i < NUM_SENSORS; i++)
    {
        classification_subs[i] = node->create_subscription<lasr_msgs::msg::ContactClassification>(
            "/d360/classified/index_" + std::to_string(i), 1,
            [i](const lasr_msgs::msg::ContactClassification::SharedPtr msg)
            {
                classification_out[i] = *msg;
                valid_flags |= S_CLASSIF[i];
            });
    }

    // Indexed audio subscriptions (2 channels per sensor)
    std::array<std::array<rclcpp::Subscription<d360_msgs::msg::AudioDataD360>::SharedPtr, 2>, NUM_SENSORS> audio_subs;
    for (int i = 0; i < NUM_SENSORS; i++)
    {
        audio_subs[i][0]
            = node->create_subscription<d360_msgs::msg::AudioDataD360>("/mic_0/index_" + std::to_string(i), 1,
                [i](const d360_msgs::msg::AudioDataD360::SharedPtr msg)
                {
                    audioOut[i].channel_0 = msg->data;
                    audioOut[i].sample_rate = msg->sample_rate;
                });
        audio_subs[i][1]
            = node->create_subscription<d360_msgs::msg::AudioDataD360>("/mic_1/index_" + std::to_string(i), 1,
                [i](const d360_msgs::msg::AudioDataD360::SharedPtr msg)
                {
                    audioOut[i].channel_1 = msg->data;
                    audioOut[i].sample_rate = msg->sample_rate;
                    valid_flags |= S_AUDIO[i];
                });
    }

    // auto digitRGB_subs = ...;
    // for (int i = 0; i < NUM_SENSORS; i++)
    // {
    //     digitRGB_subs[i] = node->create_subscription<sensor_msgs::msg::Image>(
    //         "/digit_rgb/index_" + std::to_string(i), 1,
    //         [i](const sensor_msgs::msg::Image::SharedPtr msg) {
    //             digitRGB_out[i] = *msg;
    //         });
    // }

    // --- Timer ---
    publishTimer = node->create_wall_timer(std::chrono::milliseconds(PUBLISH_FREQUENCY), publish);

    // --- Publishers ---
    qposPublisher = node->create_publisher<lasr_msgs::msg::Qpos>("/lasr/qpos", 1);
    qvelPublisher = node->create_publisher<lasr_msgs::msg::Qvel>("/lasr/qvel", 1);
    actionPublisher = node->create_publisher<lasr_msgs::msg::Action>("/lasr/action", 1);
    actionPublisherEE = node->create_publisher<lasr_msgs::msg::Action>("/lasr/action/ee", 1);
    realsenseImagePublisher = node->create_publisher<sensor_msgs::msg::Image>("/lasr/realsense/image", 1);
    realsenseDepthImagePublisher = node->create_publisher<sensor_msgs::msg::Image>("/lasr/realsense/image_depth", 1);

    for (int i = 0; i < NUM_SENSORS; i++)
    {
        auto idx = std::to_string(i);
        digit360_publishers[i] = node->create_publisher<sensor_msgs::msg::Image>("/lasr/digit360/image_" + idx, 1);
        classification_publishers[i] = node->create_publisher<lasr_msgs::msg::ContactClassification>(
            "/lasr/digit360/classification_" + idx, 1);
        digitRGB_publishers[i] = node->create_publisher<sensor_msgs::msg::Image>("/lasr/digit/rgb_image_" + idx, 1);
        audio_publishers[i] = node->create_publisher<lasr_msgs::msg::Audio>("/lasr/audio/index_" + idx, 1);
    }

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 1;
}
