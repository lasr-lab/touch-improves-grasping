#include "tilburg_hand_interface/motor_controller.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace tilburg_hand_interface
{

MotorController::MotorController(const std::string& device_name, const std::string& hand_side)
    : device_name_(device_name)
{
    initializeCalibration(hand_side);

    // Initialize Dynamixel SDK objects
    port_handler_
        = std::unique_ptr<dynamixel::PortHandler>(dynamixel::PortHandler::getPortHandler(device_name_.c_str()));
    packet_handler_
        = std::unique_ptr<dynamixel::PacketHandler>(dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION));
}

MotorController::~MotorController() { shutdown(); }

void MotorController::initializeCalibration(const std::string& hand_side)
{
    if (hand_side == "right")
    {
        calibration_.motor_calib_min_range_deg = { -5, 0, 0, 0, -5, -5, 0, -25, -5, -5, 0, -25, -5, -5, 0, -25 };
        calibration_.motor_calib_max_range_deg = { 95, 90, 100, 90, 95, 95, 95, 25, 95, 95, 95, 25, 95, 95, 95, 25 };
        calibration_.motor_calib_min_range_ticks
            = { 1991, 2048, 2048, 2048, 1991, 1991, 2048, 2332, 1991, 1991, 2048, 2332, 1991, 1991, 2048, 2332 };
        calibration_.motor_calib_max_range_ticks
            = { 3128, 1024, 910, 3072, 3128, 3128, 967, 1763, 3128, 3128, 967, 1763, 3128, 3128, 967, 1763 };
    }
    else if (hand_side == "left")
    {
        calibration_.motor_calib_min_range_deg = { -5, 0, 0, 0, -5, -5, 0, -25, -5, -5, 0, -25, -5, -5, 0, -25 };
        calibration_.motor_calib_max_range_deg = { 95, 90, 100, 90, 95, 95, 95, 25, 95, 95, 95, 25, 95, 95, 95, 25 };
        calibration_.motor_calib_min_range_ticks
            = { 1991, 2048, 2048, 2048, 1991, 1991, 2048, 1763, 1991, 1991, 2048, 1763, 1991, 1991, 2048, 1763 };
        calibration_.motor_calib_max_range_ticks
            = { 3128, 1024, 3185, 1024, 3128, 3128, 967, 2332, 3128, 3128, 967, 2332, 3128, 3128, 967, 2332 };
    }
    else
    {
        throw std::invalid_argument("Hand side must be 'left' or 'right'");
    }

    calibration_.motor_calib_zero_pos_ticks
        = { 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048 };
    calibration_.hand_orientation = hand_side;
}

bool MotorController::initialize()
{
    // Open port
    if (!port_handler_->openPort())
    {
        std::cerr << "Failed to open port " << device_name_ << std::endl;
        return false;
    }

    // Set baudrate
    if (!port_handler_->setBaudRate(BAUDRATE))
    {
        std::cerr << "Failed to set baudrate to " << BAUDRATE << std::endl;
        return false;
    }

    // Increase current limit to reduce overload protection sensitivity
    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
        uint8_t motor_id = joint_motor_mapping_[i];
        uint16_t current_limit = 0;
        uint8_t dxl_error = 0;

        int dxl_comm_result = packet_handler_->read2ByteTxRx(
            port_handler_.get(), motor_id, ADDRESS_CURRENT_LIMIT, &current_limit, &dxl_error);
        if (dxl_comm_result != COMM_SUCCESS)
        {
            std::cerr << "Failed to read current limit for motor " << static_cast<int>(motor_id) << std::endl;
            return false;
        }

        uint16_t new_limit = static_cast<uint16_t>(current_limit * 2);
        dxl_comm_result = packet_handler_->write2ByteTxRx(
            port_handler_.get(), motor_id, ADDRESS_CURRENT_LIMIT, new_limit, &dxl_error);
        if (dxl_comm_result != COMM_SUCCESS)
        {
            std::cerr << "Failed to write current limit for motor " << static_cast<int>(motor_id) << std::endl;
            return false;
        }

        std::cout << "Motor " << static_cast<int>(motor_id)
                  << " current limit: " << current_limit << " -> " << new_limit << std::endl;
    }

    // Initialize GroupSyncWrite for goal position
    group_sync_write_ = std::make_unique<dynamixel::GroupSyncWrite>(
        port_handler_.get(), packet_handler_.get(), ADDRESS_GOAL_POSITION, 4);

    // Initialize single GroupSyncRead for current, velocity, and position (10 bytes total)
    // Starting from ADDRESS_PRESENT_CURRENT (126), reads current (2 bytes) + velocity (4 bytes) + position (4 bytes)
    group_sync_read_all_ = std::make_unique<dynamixel::GroupSyncRead>(
        port_handler_.get(), packet_handler_.get(), ADDRESS_PRESENT_CURRENT, LEN_GROUP_READ_ALL);

    // Add motors to group sync read
    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
        uint8_t motor_id = joint_motor_mapping_[i];

        if (!group_sync_read_all_->addParam(motor_id))
        {
            std::cerr << "Failed to add motor " << static_cast<int>(motor_id) << " to group read" << std::endl;
            return false;
        }
    }

    std::cout << "Motor controller initialized successfully for " << calibration_.hand_orientation << " hand"
              << std::endl;
    return true;
}

void MotorController::shutdown()
{
    if (port_handler_ && port_handler_->is_using_)
    {
        disableTorque();
        port_handler_->closePort();
    }
}

bool MotorController::enableTorqueAndLED()
{
    int dxl_comm_result;
    uint8_t dxl_error = 0;

    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
        uint8_t motor_id = joint_motor_mapping_[i];

        // Enable torque
        dxl_comm_result
            = packet_handler_->write1ByteTxRx(port_handler_.get(), motor_id, ADDRESS_TORQUE_ENABLE, 1, &dxl_error);
        if (!checkCommResult(dxl_comm_result, dxl_error, "enable torque for motor " + std::to_string(motor_id)))
        {
            return false;
        }

        // Turn on LED
        dxl_comm_result = packet_handler_->write1ByteTxRx(port_handler_.get(), motor_id, ADDRESS_LED, 1, &dxl_error);
        if (!checkCommResult(dxl_comm_result, dxl_error, "turn on LED for motor " + std::to_string(motor_id)))
        {
            return false;
        }
    }

    std::cout << "Torque and LEDs enabled for all motors" << std::endl;
    return true;
}

bool MotorController::disableTorque()
{
    int dxl_comm_result;
    uint8_t dxl_error = 0;

    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
        uint8_t motor_id = joint_motor_mapping_[i];

        // Disable torque
        dxl_comm_result
            = packet_handler_->write1ByteTxRx(port_handler_.get(), motor_id, ADDRESS_TORQUE_ENABLE, 0, &dxl_error);
        checkCommResult(dxl_comm_result, dxl_error, "disable torque for motor " + std::to_string(motor_id));

        // Turn off LED
        dxl_comm_result = packet_handler_->write1ByteTxRx(port_handler_.get(), motor_id, ADDRESS_LED, 0, &dxl_error);
        checkCommResult(dxl_comm_result, dxl_error, "turn off LED for motor " + std::to_string(motor_id));
    }

    return true;
}

bool MotorController::setJointPositions(const std::vector<double>& joint_positions_deg)
{
    if (joint_positions_deg.size() != NUM_JOINTS)
    {
        std::cerr << "Expected " << NUM_JOINTS << " joint positions, got " << joint_positions_deg.size() << std::endl;
        return false;
    }

    // Clear previous parameters
    group_sync_write_->clearParam();

    // Add goal positions for all motors
    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
        uint8_t motor_id = joint_motor_mapping_[i];
        int32_t goal_position_ticks = degreesToTicks(joint_positions_deg[i], i);

        uint8_t param_goal_position[4];
        param_goal_position[0] = (uint8_t)(goal_position_ticks & 0xFF);
        param_goal_position[1] = (uint8_t)((goal_position_ticks >> 8) & 0xFF);
        param_goal_position[2] = (uint8_t)((goal_position_ticks >> 16) & 0xFF);
        param_goal_position[3] = (uint8_t)((goal_position_ticks >> 24) & 0xFF);

        if (!group_sync_write_->addParam(motor_id, param_goal_position))
        {
            std::cerr << "Failed to add parameter for motor " << static_cast<int>(motor_id) << std::endl;
            return false;
        }
    }

    // Transmit all goal positions at once
    int dxl_comm_result = group_sync_write_->txPacket();
    if (dxl_comm_result != COMM_SUCCESS)
    {
        std::cerr << "GroupSyncWrite failed: " << packet_handler_->getTxRxResult(dxl_comm_result) << std::endl;
        return false;
    }

    return true;
}

bool MotorController::readJointStates(
    std::vector<double>& positions_deg, std::vector<double>& velocities_deg_per_sec, std::vector<double>& currents_ma)
{
    positions_deg.resize(NUM_JOINTS);
    velocities_deg_per_sec.resize(NUM_JOINTS);
    currents_ma.resize(NUM_JOINTS);

    // Read all data (current, velocity, position) in a single group read
    int dxl_comm_result = group_sync_read_all_->txRxPacket();
    if (dxl_comm_result != COMM_SUCCESS)
    {
        std::cerr << "GroupSyncRead failed: " << packet_handler_->getTxRxResult(dxl_comm_result) << std::endl;
        return false;
    }

    // Extract data for all motors
    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
        uint8_t motor_id = joint_motor_mapping_[i];

        // Check if data is available
        if (!group_sync_read_all_->isAvailable(motor_id, ADDRESS_PRESENT_CURRENT, LEN_GROUP_READ_ALL))
        {
            std::cerr << "Data not available for motor " << static_cast<int>(motor_id) << std::endl;
            return false;
        }

        // Extract current (2 bytes from offset 0)
        int16_t present_current = static_cast<int16_t>(
            group_sync_read_all_->getData(motor_id, ADDRESS_PRESENT_CURRENT, LEN_PRESENT_CURRENT));
        currents_ma[i] = currentTicksToMilliamps(present_current);

        // Extract velocity (4 bytes from offset 2)
        int32_t present_velocity = static_cast<int32_t>(
            group_sync_read_all_->getData(motor_id, ADDRESS_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY));
        velocities_deg_per_sec[i] = velocityTicksToDegreesPerSec(present_velocity);

        // Extract position (4 bytes from offset 6)
        int32_t present_position = static_cast<int32_t>(
            group_sync_read_all_->getData(motor_id, ADDRESS_PRESENT_POSITION, LEN_PRESENT_POSITION));
        positions_deg[i] = ticksToDegrees(present_position, i);
    }

    return true;
}

int32_t MotorController::degreesToTicks(double degrees, size_t joint_index) const
{
    if (joint_index >= NUM_JOINTS)
    {
        throw std::out_of_range("Joint index out of range");
    }

    double min_deg = calibration_.motor_calib_min_range_deg[joint_index];
    double max_deg = calibration_.motor_calib_max_range_deg[joint_index];
    int32_t min_ticks = calibration_.motor_calib_min_range_ticks[joint_index];
    int32_t max_ticks = calibration_.motor_calib_max_range_ticks[joint_index];

    // Clamp degrees to valid range
    degrees = std::max(min_deg, std::min(max_deg, degrees));

    // Linear interpolation
    double ratio = (degrees - min_deg) / (max_deg - min_deg);
    int32_t ticks = static_cast<int32_t>(min_ticks + ratio * (max_ticks - min_ticks));

    return ticks;
}

double MotorController::ticksToDegrees(int32_t ticks, size_t joint_index) const
{
    if (joint_index >= NUM_JOINTS)
    {
        throw std::out_of_range("Joint index out of range");
    }

    double min_deg = calibration_.motor_calib_min_range_deg[joint_index];
    double max_deg = calibration_.motor_calib_max_range_deg[joint_index];
    int32_t min_ticks = calibration_.motor_calib_min_range_ticks[joint_index];
    int32_t max_ticks = calibration_.motor_calib_max_range_ticks[joint_index];

    // Linear interpolation
    double ratio = static_cast<double>(ticks - min_ticks) / (max_ticks - min_ticks);
    double degrees = min_deg + ratio * (max_deg - min_deg);

    return degrees;
}

double MotorController::velocityTicksToDegreesPerSec(int32_t velocity_ticks) const
{
    // Dynamixel velocity unit: 0.229 RPM per unit
    // Convert to degrees per second: RPM * 6.0 = degrees per second
    double rpm = velocity_ticks * 0.229;
    return rpm * 6.0;
}

double MotorController::currentTicksToMilliamps(int16_t current_ticks) const
{
    // Dynamixel current unit: 2.69 mA per unit (for XM430 series)
    return current_ticks * 2.69;
}

bool MotorController::checkCommResult(int dxl_comm_result, uint8_t dxl_error, const std::string& operation)
{
    if (dxl_comm_result != COMM_SUCCESS)
    {
        std::cerr << "Failed to " << operation << ": " << packet_handler_->getTxRxResult(dxl_comm_result) << std::endl;
        return false;
    }

    if (dxl_error != 0)
    {
        std::cerr << "Dynamixel error during " << operation << ": " << packet_handler_->getRxPacketError(dxl_error)
                  << std::endl;
        return false;
    }

    return true;
}

} // namespace tilburg_hand_interface
