#ifndef TILBURG_HAND_INTERFACE__MOTOR_CONTROLLER_HPP_
#define TILBURG_HAND_INTERFACE__MOTOR_CONTROLLER_HPP_

#include "dynamixel_sdk/dynamixel_sdk.h"
#include <memory>
#include <string>
#include <vector>

namespace tilburg_hand_interface
{

struct HandCalibration
{
    std::vector<double> motor_calib_min_range_deg;
    std::vector<double> motor_calib_max_range_deg;
    std::vector<int32_t> motor_calib_min_range_ticks;
    std::vector<int32_t> motor_calib_max_range_ticks;
    std::vector<int32_t> motor_calib_zero_pos_ticks;
    std::string hand_orientation;
};

class MotorController
{
public:
    MotorController(const std::string& device_name, const std::string& hand_side);
    ~MotorController();

    bool initialize();
    void shutdown();

    bool setJointPositions(const std::vector<double>& joint_positions_deg);
    bool readJointStates(std::vector<double>& positions_deg, std::vector<double>& velocities_deg_per_sec,
        std::vector<double>& currents_ma);

    bool enableTorqueAndLED();
    bool disableTorque();

private:
    // Constants
    static constexpr double PROTOCOL_VERSION = 2.0;
    static constexpr int BAUDRATE = 4000000;
    static constexpr uint16_t ADDRESS_CURRENT_LIMIT = 38;
    static constexpr uint16_t ADDRESS_TORQUE_ENABLE = 64;
    static constexpr uint16_t ADDRESS_LED = 65;
    static constexpr uint16_t ADDRESS_GOAL_CURRENT = 102;
    static constexpr uint16_t ADDRESS_GOAL_POSITION = 116;
    static constexpr uint16_t ADDRESS_PRESENT_CURRENT = 126;
    static constexpr uint16_t ADDRESS_PRESENT_VELOCITY = 128;
    static constexpr uint16_t ADDRESS_PRESENT_POSITION = 132;
    static constexpr uint16_t ADDRESS_PRESENT_TEMPERATURE = 146;

    // Data lengths for group read
    static constexpr uint16_t LEN_PRESENT_CURRENT = 2; // 2 bytes
    static constexpr uint16_t LEN_PRESENT_VELOCITY = 4; // 4 bytes
    static constexpr uint16_t LEN_PRESENT_POSITION = 4; // 4 bytes
    static constexpr uint16_t LEN_GROUP_READ_ALL = 10; // Total: 2 + 4 + 4 = 10 bytes

    static constexpr size_t NUM_JOINTS = 16;

    // Motor IDs
    const std::vector<uint8_t> joint_motor_mapping_
        = { 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28, 35, 34, 33, 32 };

    // Conversion functions
    int32_t degreesToTicks(double degrees, size_t joint_index) const;
    double ticksToDegrees(int32_t ticks, size_t joint_index) const;
    double velocityTicksToDegreesPerSec(int32_t velocity_ticks) const;
    double currentTicksToMilliamps(int16_t current_ticks) const;

    // Dynamixel SDK objects
    std::unique_ptr<dynamixel::PortHandler> port_handler_;
    std::unique_ptr<dynamixel::PacketHandler> packet_handler_;
    std::unique_ptr<dynamixel::GroupSyncWrite> group_sync_write_;
    std::unique_ptr<dynamixel::GroupSyncRead> group_sync_read_all_; // Single group read for all data

    // Configuration
    std::string device_name_;
    HandCalibration calibration_;

    // Helper functions
    void initializeCalibration(const std::string& hand_side);
    bool checkCommResult(int dxl_comm_result, uint8_t dxl_error, const std::string& operation);
};

} // namespace tilburg_hand_interface

#endif // TILBURG_HAND_INTERFACE__MOTOR_CONTROLLER_HPP_
