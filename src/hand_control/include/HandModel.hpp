#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "lasr_msgs/msg/robot_hand.hpp"

struct Dimension
{
    Dimension() { }

    Dimension(Eigen::Vector2d depth, Eigen::Vector2d width, Eigen::Vector2d height)
        : depth(depth)
        , width(width)
        , height(height)
    {
    }

    Eigen::Vector2d getDepth() { return this->depth; }
    Eigen::Vector2d getWidth() { return this->width; }
    Eigen::Vector2d getHeight() { return this->height; }

    Eigen::Vector2d depth;
    Eigen::Vector2d width;
    Eigen::Vector2d height;
};

struct BoundingBox
{
public:
    BoundingBox() { }

    BoundingBox(Dimension dimension, Eigen::Matrix4d* globalTransformation, Eigen::Vector3d rotations)
    {
        Eigen::Vector2d depth = dimension.getDepth();
        Eigen::Vector2d width = dimension.getWidth();
        Eigen::Vector2d height = dimension.getHeight();

        double angle = rotations[0] / 180.0 * M_PI;
        Eigen::Matrix4d xRotMatrix;
        xRotMatrix << std::cos(angle), std::sin(angle), 0.0, 0.0, -std::sin(angle), std::cos(angle), 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

        angle = rotations[1] / 180.0 * M_PI;
        Eigen::Matrix4d yRotMatrix;
        yRotMatrix << 1.0, 0.0, 0.0, 0.0, 0.0, std::cos(angle), std::sin(angle), 0.0, 0.0, -std::sin(angle),
            std::cos(angle), 0.0, 0.0, 0.0, 0.0, 1.0;

        angle = -rotations[2] / 180.0 * M_PI;
        Eigen::Matrix4d zRotMatrix;
        zRotMatrix << std::cos(angle), 0.0, -std::sin(angle), 0.0, 0.0, 1.0, 0.0, 0.0, std::sin(angle), 0.0,
            std::cos(angle), 0.0, 0.0, 0.0, 0.0, 1.0;

        // Lower square
        boundingBox.push_back(Eigen::Vector4d(depth[1], width[1], height[0], 1.0));
        boundingBox.push_back(Eigen::Vector4d(depth[1], width[0], height[0], 1.0));
        boundingBox.push_back(Eigen::Vector4d(depth[0], width[0], height[0], 1.0));
        boundingBox.push_back(Eigen::Vector4d(depth[0], width[1], height[0], 1.0));

        // Upper square
        boundingBox.push_back(Eigen::Vector4d(depth[0], width[1], height[1], 1.0));
        boundingBox.push_back(Eigen::Vector4d(depth[1], width[1], height[1], 1.0));
        boundingBox.push_back(Eigen::Vector4d(depth[1], width[0], height[1], 1.0));
        boundingBox.push_back(Eigen::Vector4d(depth[0], width[0], height[1], 1.0));

        Eigen::Matrix4d localRotation = yRotMatrix * (xRotMatrix * zRotMatrix);
        for (auto& point : this->boundingBox)
        {
            point = (*globalTransformation) * (localRotation * point);
        }

        rotationPoint = vec4ToVec3((*globalTransformation) * Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));

        *globalTransformation = (*globalTransformation * localRotation);
    }

    Eigen::Vector3d vec4ToVec3(Eigen::Vector4d v) { return Eigen::Vector3d(v[0], v[1], v[2]); }

    int createLineMarkers(std::vector<visualization_msgs::msg::Marker>* _outArray, int id)
    {
        auto out = visualization_msgs::msg::Marker();

        out.type = visualization_msgs::msg::Marker::LINE_STRIP;
        out.action = visualization_msgs::msg::Marker::ADD;
        out.id = id;

        std::vector<geometry_msgs::msg::Point> points;
        for (auto& point : this->boundingBox)
        {
            auto geomPoint = geometry_msgs::msg::Point();
            auto point3 = vec4ToVec3(point);
            geomPoint.x = point3[0];
            geomPoint.y = point3[1];
            geomPoint.z = point3[2];
            points.push_back(geomPoint);
        }
        std::vector<geometry_msgs::msg::Point> out_points;

        for (int i = 0; i < 8; i++)
        {
            if (i == 4)
            {
                out_points.push_back(points.at(0));
                out_points.push_back(points.at(3));
            }
            if (i == 6)
            {
                out_points.push_back(points.at(0));
                out_points.push_back(points.at(5));
            }
            if (i == 7)
            {
                out_points.push_back(points.at(1));
                out_points.push_back(points.at(6));
            }
            out_points.push_back(points.at(i));
        }
        out_points.push_back(points.at(2));
        out_points.push_back(points.at(7));
        out_points.push_back(points.at(4));

        out.color.r = 0.0f;
        out.color.g = 1.0f;
        out.color.b = 0.0f;
        out.color.a = 1.0;

        out.scale.x = 0.1;
        out.scale.y = 1.0;
        out.scale.z = 1.0;

        out.points = out_points;

        _outArray->push_back(out);

        // Rotation point

        auto outRotation = visualization_msgs::msg::Marker();

        outRotation.type = visualization_msgs::msg::Marker::SPHERE;
        outRotation.action = visualization_msgs::msg::Marker::ADD;
        outRotation.id = id + 1;

        auto rotationPose = geometry_msgs::msg::Pose();
        auto rotationPosePoint = geometry_msgs::msg::Point();

        rotationPosePoint.x = rotationPoint[0];
        rotationPosePoint.y = rotationPoint[1];
        rotationPosePoint.z = rotationPoint[2];

        rotationPose.position = rotationPosePoint;
        outRotation.pose = rotationPose;

        outRotation.scale.x = 0.5;
        outRotation.scale.y = 0.5;
        outRotation.scale.z = 0.5;

        outRotation.color.r = 1.0f;
        outRotation.color.g = 0.0f;
        outRotation.color.b = 0.0f;
        outRotation.color.a = 1.0;

        _outArray->push_back(outRotation);

        return id + 2;
    }

    std::vector<Eigen::Vector4d> boundingBox;
    Eigen::Vector3d rotationPoint;
};

struct Finger
{
public:
    Finger() { }

    Finger(double fingerTip, double fingerFirstJoint, double fingerBaseJoint, double fingerBaseRot,
        Eigen::Vector3d fingerOffset, Eigen::Vector3d baseOffset)
        : fingerTip(fingerTip)
        , fingerFirstJoint(fingerFirstJoint)
        , fingerBaseJoint(fingerBaseJoint)
        , fingerBaseRot(fingerBaseRot)
        , fingerOffset(fingerOffset)
        , baseOffset(baseOffset)
    {
    }

    virtual ~Finger() = default;

    virtual void createBoundingBox()
    {
        this->initBaseSegmentRotation();
        this->initFingerDimensions();
        this->initBoundingBox();
    }

    virtual void initBaseSegmentRotation() { this->baseSegmentRotation << 0.0, fingerBaseRot, fingerBaseJoint; }

    virtual void initFingerDimensions()
    {
        this->baseBoxDimension
            = Dimension(Eigen::Vector2d(1.15, -1.15), Eigen::Vector2d(1.45, -1.45), Eigen::Vector2d(-0.85, 5.95));
        this->firstBoxDimension
            = Dimension(Eigen::Vector2d(1.15, -1.15), Eigen::Vector2d(1.45, -1.45), Eigen::Vector2d(1.6, 5.2));
        this->tipBoxDimension
            = Dimension(Eigen::Vector2d(1.3, -2.0), Eigen::Vector2d(1.55, -1.55), Eigen::Vector2d(1.6, 5.1));
        this->firstOffset = Eigen::Vector3d(0.0, 0.0, 5.0);
        this->tipOffset = Eigen::Vector3d(0.0, 0.0, 4.3);
    }

    void initBoundingBox()
    {
        Eigen::Matrix4d globalRotation;
        globalRotation << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

        Eigen::Matrix4d handTransformation;
        handTransformation << 1.0, 0.0, 0.0, fingerOffset[0], 0.0, 1.0, 0.0, fingerOffset[1], 0.0, 0.0, 1.0,
            fingerOffset[2], 0.0, 0.0, 0.0, 1.0;

        Eigen::Matrix4d baseTransformationBase;
        baseTransformationBase << 1.0, 0.0, 0.0, baseOffset[0], 0.0, 1.0, 0.0, baseOffset[1], 0.0, 0.0, 1.0,
            baseOffset[2], 0.0, 0.0, 0.0, 1.0;

        Eigen::Matrix4d globalTransformation;
        globalTransformation = baseTransformationBase * (globalRotation * handTransformation);
        baseBox = BoundingBox(baseBoxDimension, &globalTransformation, baseSegmentRotation);

        Eigen::Matrix4d baseTransformationFirst;
        baseTransformationFirst << 1.0, 0.0, 0.0, firstOffset[0], 0.0, 1.0, 0.0, firstOffset[1], 0.0, 0.0, 1.0,
            firstOffset[2], 0.0, 0.0, 0.0, 1.0;
        globalTransformation = globalTransformation * baseTransformationFirst;
        firstBox = BoundingBox(firstBoxDimension, &globalTransformation, Eigen::Vector3d(0.0, 0.0, fingerFirstJoint));

        Eigen::Matrix4d baseTransformationTip;
        baseTransformationTip << 1.0, 0.0, 0.0, tipOffset[0], 0.0, 1.0, 0.0, tipOffset[1], 0.0, 0.0, 1.0, tipOffset[2],
            0.0, 0.0, 0.0, 1.0;
        globalTransformation = globalTransformation * baseTransformationTip;
        tipBox = BoundingBox(tipBoxDimension, &globalTransformation, Eigen::Vector3d(0.0, 0.0, fingerTip));
    }

    int createVisMessage(std::vector<visualization_msgs::msg::Marker>* _outArray, int base_id)
    {
        base_id = baseBox.createLineMarkers(_outArray, base_id);
        base_id = firstBox.createLineMarkers(_outArray, base_id);
        base_id = tipBox.createLineMarkers(_outArray, base_id);
        return base_id;
    }

    BoundingBox baseBox;
    BoundingBox firstBox;
    BoundingBox tipBox;

    Dimension baseBoxDimension;
    Dimension firstBoxDimension;
    Dimension tipBoxDimension;

    double fingerTip = 0.0;
    double fingerFirstJoint = 0.0;
    double fingerBaseJoint = 0.0;
    double fingerBaseRot = 0.0;

    Eigen::Vector3d fingerOffset;
    Eigen::Vector3d baseOffset;
    Eigen::Vector3d firstOffset;
    Eigen::Vector3d tipOffset;

    Eigen::Vector3d baseSegmentRotation;
};

struct Thumb : public Finger
{
    using Finger::Finger;

    void initBaseSegmentRotation() override { this->baseSegmentRotation << fingerBaseRot, 90.0, fingerBaseJoint; }

    void initFingerDimensions() override
    {
        this->baseBoxDimension
            = Dimension(Eigen::Vector2d(1.0, -2.8), Eigen::Vector2d(1.35, -1.35), Eigen::Vector2d(-0.8, 1.9));
        this->firstBoxDimension
            = Dimension(Eigen::Vector2d(1.15, -1.15), Eigen::Vector2d(1.45, -1.45), Eigen::Vector2d(-0.85, 5.95));
        this->tipBoxDimension
            = Dimension(Eigen::Vector2d(1.3, -2.0), Eigen::Vector2d(1.55, -1.55), Eigen::Vector2d(1.6, 5.1));
        this->firstOffset = Eigen::Vector3d(0.0, 0.0, 3.7);
        this->tipOffset = Eigen::Vector3d(0.0, 0.0, 5.0);
    }

    void createBoundingBox() override
    {
        this->initBaseSegmentRotation();
        this->initFingerDimensions();
        this->initBoundingBox();
    }
};

struct Hand
{
public:
    Hand(Thumb t, Finger i, Finger m, Finger r)
        : thumb(std::make_unique<Thumb>(std::move(t)))
        , index(std::make_unique<Finger>(std::move(i)))
        , middle(std::make_unique<Finger>(std::move(m)))
        , ring(std::make_unique<Finger>(std::move(r)))
    {
        this->thumb->createBoundingBox();
        this->index->createBoundingBox();
        this->middle->createBoundingBox();
        this->ring->createBoundingBox();
    }

    explicit Hand(const lasr_msgs::msg::RobotHand& msg)
    {
        this->thumb = std::make_unique<Thumb>(msg.thumb.finger_tip, msg.thumb.finger_first_joint,
            msg.thumb.finger_base_joint, msg.thumb.finger_base_rot, thumbFingerOffset, thumbBaseOffset);
        this->index = std::make_unique<Finger>(msg.index_finger.finger_tip, msg.index_finger.finger_first_joint,
            msg.index_finger.finger_base_joint, msg.index_finger.finger_base_rot, indexFingerOffset, indexBaseOffset);
        this->middle = std::make_unique<Finger>(msg.middle_finger.finger_tip, msg.middle_finger.finger_first_joint,
            msg.middle_finger.finger_base_joint, msg.middle_finger.finger_base_rot, middleFingerOffset,
            middleBaseOffset);
        this->ring = std::make_unique<Finger>(msg.ring_finger.finger_tip, msg.ring_finger.finger_first_joint,
            msg.ring_finger.finger_base_joint, msg.ring_finger.finger_base_rot, ringFingerOffset, ringBaseOffset);

        this->thumb->createBoundingBox();
        this->index->createBoundingBox();
        this->middle->createBoundingBox();
        this->ring->createBoundingBox();
    }

    Hand(Hand&&) = default;
    Hand& operator=(Hand&&) = default;
    Hand(const Hand&) = delete;
    Hand& operator=(const Hand&) = delete;

    ~Hand() = default;

    bool isColliding() { return false; }

    void fillVisMessage(std::vector<visualization_msgs::msg::Marker>* markerVector)
    {
        int base_id = 0;
        base_id = this->thumb->createVisMessage(markerVector, base_id);
        base_id = this->index->createVisMessage(markerVector, base_id);
        base_id = this->middle->createVisMessage(markerVector, base_id);
        base_id = this->ring->createVisMessage(markerVector, base_id);
    }

    std::unique_ptr<Thumb> thumb;
    std::unique_ptr<Finger> index;
    std::unique_ptr<Finger> middle;
    std::unique_ptr<Finger> ring;

    Eigen::Vector3d thumbBaseOffset = Eigen::Vector3d(0.0, 1.5, 0.0);
    Eigen::Vector3d indexBaseOffset = Eigen::Vector3d(0.0, 0.0, 2.9);
    Eigen::Vector3d middleBaseOffset = Eigen::Vector3d(0.0, 0.0, 2.9);
    Eigen::Vector3d ringBaseOffset = Eigen::Vector3d(0.0, 0.0, 2.9);

    Eigen::Vector3d thumbFingerOffset = Eigen::Vector3d(0.0, 4.7, 0.0);
    Eigen::Vector3d indexFingerOffset = Eigen::Vector3d(0.0, 4.4, 7.1);
    Eigen::Vector3d middleFingerOffset = Eigen::Vector3d(0.0, 0.0, 7.3);
    Eigen::Vector3d ringFingerOffset = Eigen::Vector3d(0.0, -4.4, 7.2);
};

struct HandModel
{
public:
    explicit HandModel(Hand currentHand)
        : currentHand(std::move(currentHand))
    {
    }

    bool update(Hand request)
    {
        if (this->isColliding(request))
        {
            return false;
        }
        this->currentHand = std::move(request);
        return true;
    }

    void handMessageFromModel(std_msgs::msg::Float64MultiArray& out)
    {
        out.data = { this->currentHand.thumb->fingerTip, this->currentHand.thumb->fingerFirstJoint,
            this->currentHand.thumb->fingerBaseRot, this->currentHand.thumb->fingerBaseJoint,
            this->currentHand.index->fingerTip, this->currentHand.index->fingerFirstJoint,
            this->currentHand.index->fingerBaseJoint, this->currentHand.index->fingerBaseRot,
            this->currentHand.middle->fingerTip, this->currentHand.middle->fingerFirstJoint,
            this->currentHand.middle->fingerBaseJoint, this->currentHand.middle->fingerBaseRot,
            this->currentHand.ring->fingerTip, this->currentHand.ring->fingerFirstJoint,
            this->currentHand.ring->fingerBaseJoint, this->currentHand.ring->fingerBaseRot };
    }

    void visMessageFromModel(std::vector<visualization_msgs::msg::Marker>* markerVector)
    {
        this->currentHand.fillVisMessage(markerVector);
    }

    ~HandModel() = default;

private:
    bool isColliding(Hand& request) { return request.isColliding(); }
    Hand currentHand;
};
