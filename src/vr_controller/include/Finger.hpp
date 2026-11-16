#include <Eigen/Dense>

struct Finger
{
    Eigen::Vector3d firstJointPosition;
    Eigen::Vector3d secondJointPosition;
    Eigen::Vector3d tipPosition;
};