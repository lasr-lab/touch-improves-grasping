using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using TMPro;
using RosMessageTypes.Lasr;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using System;
using Oculus.Haptics;
using UnityEngine.UI;


public class Tracking : MonoBehaviour
{
    ROSConnection ros;
    public string pubTopicName = "/xarm/command";
    public string handTopicName = "/vr/hand/request";
    public string subWatchdog = "/watchdog/status";
    public string cameraStreamTopicName = "/camera/camera/color/image_raw";
    public string captureTopicName = "capture";
    public string processBagServiceName = "process_bag";

    public TextMeshPro axisText;
    public TextMeshPro recordText;
    public TextMeshPro statusText;
    public TextMeshPro processingText;

    public GameObject controllerTracker;
    public GameObject handTracker;

    public GameObject thumbTipTracker;
    public GameObject thumbFirstTracker;
    public GameObject thumbSecondTracker;
    public GameObject thumbBaseTracker;
    public GameObject thumbEETracker;

    public GameObject indexTipTracker;
    public GameObject indexFirstTracker;
    public GameObject indexSecondTracker;
    public GameObject indexBaseTracker;
    public GameObject indexEETracker;


    public GameObject middleTipTracker;
    public GameObject middleFirstTracker;
    public GameObject middleSecondTracker;
    public GameObject middleBaseTracker;
    public GameObject middleEETracker;


    public GameObject ringTipTracker;
    public GameObject ringFirstTracker;
    public GameObject ringSecondTracker;
    public GameObject ringBaseTracker;
    public GameObject ringEETracker;

    public GameObject cameraImage;
    GameObject trackerObject;

    // Publish the controller's position and rotation every N seconds
    public double publishMessageFrequency = 0.01f;
    public double captureFrequency = 1.5f;
    public double FORWARD_RANGE = 0.48;
    public double HEIGHT_RANGE = 0.93;
    public double SIDE_RANGE = 0.80;
    public double MAX_ANGLE = Math.PI;
    public double MIN_SPEED = 0.0;
    public double MAX_SPEED = 1000.0;

    int currentAxis = 0;

    int fpsCounter = 0;
    bool currentlySwitching = false;
    int switchingCounter = 0;

    bool[] flipStates = { false, false, false, false, true, true };

    bool processing = false;
    bool recordingAvailable = false;
    bool currentlyCapturing;

    double globalX;
    double globalY;
    double globalZ;
    double globalTransformX;
    double globalTransformY;
    double globalTransformZ;

    double globalAngleX;
    double globalAngleY;
    double globalAngleZ;
    double globalAngleW;
    double globalRotationX;
    double globalRotationY;
    double globalRotationZ;
    double globalRotationW;

    int successCounter = 0;


    // Used to determine how much time has elapsed since the last message was published
    private double timeElapsed;
    private double captureTimeElapsed;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // start the ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<XArmCommandMsg>(pubTopicName);
        ros.RegisterPublisher<BoolMsg>(captureTopicName);
        ros.RegisterPublisher<HandPositionMsg>(handTopicName);
        ros.Subscribe<BoolMsg>(subWatchdog, updateWatchdogStatus);
        ros.Subscribe<ImageMsg>(cameraStreamTopicName, updateCameraImage);

        ros.RegisterRosService<SetBoolRequest, SetBoolResponse>(processBagServiceName);
    }

    void updateWatchdogStatus(BoolMsg msg)
    {
        if (msg.data)
        {
            statusText.text = "";
        }
        else
        {
            statusText.text = "Error";
        }
    }

    double crop(double lower, double upper, double value)
    {
        if (value <= lower)
        {
            return lower;
        }
        if (value >= upper)
        {
            return upper;
        }
        return value;
    }

    void toggleRecordingText()
    {
        if (currentlyCapturing)
        {
            recordText.text = "Rec";
        }
        else
        {
            recordText.text = "";
        }
    }

    void callbackBagRequest(SetBoolResponse response)
    {
        processing = false;
        processingText.text = "";
    }

    void sendBagRequest(bool was_success)
    {
        if (was_success)
        {
            successCounter += 1;
        }
        processingText.text = "Pro\n" + successCounter;
        SetBoolRequest request = new SetBoolRequest();
        request.data = was_success;
        ros.SendServiceMessage<SetBoolResponse>(processBagServiceName, request, callbackBagRequest);
        processing = true;
    }

    double shiftToDomain(double range, double value, double lower, double upper)
    {
        double percentage = (value + (range / 2.0)) / range;
        if (percentage < 0.0)
            percentage = 0.0;

        double out_scale = Math.Abs(upper - lower);
        double res = lower + out_scale * percentage;

        return crop(lower, upper, res);
    }

    void updateCameraImage(ImageMsg msg)
    {
        return;
        fpsCounter += 1;
        if (fpsCounter % 60 == 0)
        {
            fpsCounter = 0;
            Texture2D videotexture = new Texture2D((int)msg.width, (int)msg.height);
            cameraImage.GetComponent<RawImage>().color = new Color(255, 255, 255, 255);
            for (int y = videotexture.height - 1; y > -1; y--)
            {
                for (int x = 0; x < videotexture.width; x++) //Goes through each pixel
                {
                    int data_x = x;
                    int y_data = y;
                    Color pixelColour;
                    int base_index = y * videotexture.width * 3 + x * 3;
                    pixelColour = new Color32(msg.data[base_index], msg.data[base_index + 1], msg.data[base_index + 2], 255);
                    videotexture.SetPixel(x, videotexture.height - y - 1, pixelColour);
                }
            }
            videotexture.Apply();
            cameraImage.GetComponent<RawImage>().texture = (videotexture as Texture);

        }
    }

    void updateTracker()
    {
        if (OVRInput.IsControllerConnected(OVRInput.Controller.RHand))
        {
            trackerObject = handTracker;
        }
        else
        {
            trackerObject = controllerTracker;
        }
    }

    FingerPositionMsg assembleFingerPositionMessage(GameObject baseTracker, GameObject secondTracker, GameObject firstTracker, GameObject tipTracker)
    {
        HeaderMsg header = new HeaderMsg();

        Quaternion handRotation = Quaternion.Inverse(handTracker.transform.rotation);
        Quaternion baseRotation = Quaternion.Inverse(baseTracker.transform.rotation);
        Quaternion secondRotation = Quaternion.Inverse(secondTracker.transform.rotation);

        Vector3 basePosition = handRotation * (baseTracker.transform.position - handTracker.transform.position);

        Vector3 firstJointPos = handRotation * (secondTracker.transform.position - baseTracker.transform.position);
        Vector3Msg indexFingerFirstJointPosition = new Vector3Msg(firstJointPos.x, firstJointPos.y, firstJointPos.z);

        Vector3 secondJointPos = baseRotation * (firstTracker.transform.position - secondTracker.transform.position);
        Vector3Msg indexFingerSecondJointPosition = new Vector3Msg(secondJointPos.x, secondJointPos.y, secondJointPos.z);

        Vector3 tipPos = secondRotation * (tipTracker.transform.position - firstTracker.transform.position);
        Vector3Msg tipFingerPosition = new Vector3Msg(tipPos.x, tipPos.y, tipPos.z);

        FingerPositionMsg indexFinger = new FingerPositionMsg(header, indexFingerFirstJointPosition, indexFingerSecondJointPosition, tipFingerPosition);
        return indexFinger;
    }


    void publishHandPositionMessage(bool is_origin)
    {
        HeaderMsg header = new HeaderMsg();

        FingerPositionMsg thumb = assembleFingerPositionMessage(thumbBaseTracker, thumbSecondTracker, thumbFirstTracker, thumbTipTracker);
        FingerPositionMsg index = assembleFingerPositionMessage(indexBaseTracker, indexSecondTracker, indexFirstTracker, indexTipTracker);
        FingerPositionMsg middle = assembleFingerPositionMessage(middleBaseTracker, middleSecondTracker, middleFirstTracker, middleTipTracker);
        FingerPositionMsg ring = assembleFingerPositionMessage(ringBaseTracker, ringSecondTracker, ringFirstTracker, ringTipTracker);

        HandPositionMsg handPositionMsg = new HandPositionMsg(header, thumb, index, middle, ring, is_origin);
        ros.Publish(handTopicName, handPositionMsg);
    }

    // Update is called once per frame
    void Update()
    {
        if (!currentlyCapturing)
        {
            toggleRecordingText();
        }
        updateTracker();
        if (OVRInput.Get(OVRInput.RawButton.Y))
        {
            globalTransformX = trackerObject.transform.position.x;
            globalTransformY = trackerObject.transform.position.y;
            globalTransformZ = trackerObject.transform.position.z;
            globalX = 0.0;
            globalY = 0.0;
            globalZ = 0.0;

            globalRotationX = trackerObject.transform.rotation.x;
            globalRotationY = trackerObject.transform.rotation.y;
            globalRotationZ = trackerObject.transform.rotation.z;
            globalRotationW = trackerObject.transform.rotation.w;
            globalAngleX = 0.0;
            globalAngleY = 0.0;
            globalAngleZ = 0.0;
            globalAngleW = 0.0;
            publishHandPositionMessage(true);
        }


        timeElapsed += Time.deltaTime;
        captureTimeElapsed += Time.deltaTime;

        if (OVRInput.Get(OVRInput.RawAxis2D.LThumbstick).y > 0.5f)
        {
            if (recordingAvailable)
            {
                recordingAvailable = false;
                sendBagRequest(true);
            }
        }
        else if (OVRInput.Get(OVRInput.RawAxis2D.LThumbstick).y < -0.5f)
        {
            if (recordingAvailable)
            {
                recordingAvailable = false;
                sendBagRequest(false);
            }
        }

        double x = (trackerObject.transform.position.x - globalTransformX) + globalX;
        double y = (trackerObject.transform.position.y - globalTransformY) + globalY;
        double z = (trackerObject.transform.position.z - globalTransformZ) + globalZ;

        double angleW = 2.0 * Math.Acos((double)trackerObject.transform.rotation.w - globalRotationW);
        double angleX = ((double)trackerObject.transform.rotation.x - globalRotationX) * angleW / 5.0 + globalAngleX; //Rotation feels faster than movement, so scaling divided by 5.0
        double angleY = ((double)trackerObject.transform.rotation.y - globalRotationY) * angleW / 5.0 + globalAngleY; //Rotation feels faster than movement, so scaling divided by 5.0
        double angleZ = ((double)trackerObject.transform.rotation.z - globalRotationZ) * angleW + globalAngleZ; //Rotation feels faster than movement, so scaling divided by 5.0

        HeaderMsg header = new HeaderMsg();
        XArmCommandMsg xarmmsg = new XArmCommandMsg(
            header,
            -(float)shiftToDomain(FORWARD_RANGE, (flipStates[0] ? -1 : 1) * -z, -800.0, -200.0),  // z -> x
           (float)shiftToDomain(SIDE_RANGE, (flipStates[1] ? -1 : 1) * -x, -400, 400),           //x -> y
            (float)shiftToDomain(HEIGHT_RANGE, (flipStates[2] ? -1 : 1) * y, 160.0, 600.0),        //y -> z
            (float)crop(-MAX_ANGLE, MAX_ANGLE, (flipStates[3] ? -1 : 1) * angleZ) + (float)Math.PI, // z -> x 
           -(float)crop(-MAX_ANGLE, MAX_ANGLE, (flipStates[4] ? -1 : 1) * angleX) - (float)(Math.PI / 2), // x -> y
            -(float)crop(-MAX_ANGLE, MAX_ANGLE, (flipStates[5] ? -1 : 1) * angleY), // y -> z
            100.0f,
            500.0f,
            0.0f
        );

        globalTransformX = trackerObject.transform.position.x;
        globalTransformY = trackerObject.transform.position.y;
        globalTransformZ = trackerObject.transform.position.z;

        globalRotationX = trackerObject.transform.rotation.x;
        globalRotationY = trackerObject.transform.rotation.y;
        globalRotationZ = trackerObject.transform.rotation.z;
        globalRotationW = trackerObject.transform.rotation.w;

        if (timeElapsed > publishMessageFrequency)
        {

            // Finally send the message to server_endpoint.py running in ROS
            if (OVRInput.Get(OVRInput.RawButton.X))
            {
                globalX = x;
                globalY = y;
                globalZ = z;

                globalAngleX = angleX;
                globalAngleY = angleY;
                globalAngleZ = angleZ;
                globalAngleW = angleW;

                ros.Publish(pubTopicName, xarmmsg);
                publishHandPositionMessage(false);
            }

            timeElapsed = 0;
        }

        if (OVRInput.Get(OVRInput.RawAxis1D.LHandTrigger, OVRInput.Controller.Touch) > 0.5f)
        {
            if (!currentlySwitching)
            {
                currentAxis = (currentAxis + 1) % 6;
            }
            currentlySwitching = true;
            switch (currentAxis)
            {
                case 0:
                    {
                        axisText.text = "x_pos";
                        break;
                    }
                case 1:
                    {
                        axisText.text = "y_pos";
                        break;
                    }
                case 2:
                    {
                        axisText.text = "z_pos";
                        break;
                    }
                case 3:
                    {
                        axisText.text = "x_rot";
                        break;
                    }
                case 4:
                    {
                        axisText.text = "y_rot";
                        break;
                    }
                case 5:
                    {
                        axisText.text = "z_rot";
                        break;
                    }
                default:
                    axisText.text = "";
                    break;
            }
        }
        else
        {
            currentlySwitching = false;
        }

        if (OVRInput.Get(OVRInput.RawAxis1D.LIndexTrigger, OVRInput.Controller.Touch) > 0.5f)
        {
            switchingCounter++;
            if (switchingCounter == 1)
            {
                flipStates[currentAxis] = !flipStates[currentAxis];
            }
        }
        else
        {
            switchingCounter = 0;
        }

        if (OVRInput.Get(OVRInput.RawButton.LThumbstick) && !processing)
        {
            if (captureTimeElapsed > captureFrequency)
            {
                currentlyCapturing = !currentlyCapturing;
                captureTimeElapsed = 0;
                BoolMsg captureOut = new BoolMsg(currentlyCapturing);
                ros.Publish(captureTopicName, captureOut);
                toggleRecordingText();
                if (!currentlyCapturing)
                {
                    recordingAvailable = true;
                }
            }
        }
    }
}

