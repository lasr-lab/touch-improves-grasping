using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class UDPManager : MonoBehaviour
{
    public static UDPManager Instance { get; private set; }

    [Header("Receive Settings (from ROS)")]
    [SerializeField] private int receivePort = 8081;

    [Header("Send Settings (to ROS)")]
    [SerializeField] private string targetIP = "127.0.0.1";
    [SerializeField] private int targetPort = 8080;

    [Header("Debug")]
    [SerializeField] private bool debugMode = false;

    // UDP clients
    private UdpClient receiveClient;
    private IPEndPoint remoteEndPoint;

    // Threading
    private Thread receiveThread;
    private bool isRunning = false;
    private readonly object lockObject = new object();
    public bool IsConnected => isRunning && receiveClient != null;

    // Message queue for thread-safe processing
    private Queue<string> messageQueue = new Queue<string>();

    // Events for message handling
    public event Action<JointStateData> OnJointStateReceived;
    public event Action<ImageData> OnColorImageReceived;
    public event Action<DepthImageData> OnDepthImageReceived;
    public event Action<CameraInfoData> OnCameraInfoReceived;

    // Chunked message reconstruction
    private Dictionary<uint, ChunkedMessage> chunkedMessages = new Dictionary<uint, ChunkedMessage>();

    private class ChunkedMessage
    {
        public byte[][] chunks;
        public int receivedCount;
        public int totalChunks;
        public string msgType;
        public float timestamp;

        public ChunkedMessage(int total, string type)
        {
            chunks = new byte[total][];
            receivedCount = 0;
            totalChunks = total;
            msgType = type;
            timestamp = Time.time;
        }
    }

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
            return;
        }
    }

    void OnDestroy()
    {
        Shutdown();
    }

    void OnApplicationQuit()
    {
        Shutdown();
    }

    public void SetTargetIP(string ip)
    {
        targetIP = ip;
    }

    public string TargetIP => targetIP;

    public void StartUDP()
    {
        InitializeUDP();
    }

    public void start()
    {
        StartUDP();
    }

    private void InitializeUDP()
    {
        try
        {
            // Initialize receive client with low-latency settings
            receiveClient = new UdpClient(receivePort);
            receiveClient.Client.ReceiveBufferSize = 1024 * 1024; // 1MB buffer for burst handling
            receiveClient.Client.ReceiveTimeout = 100; // 100ms timeout for faster processing

            // Start receive thread with high priority
            isRunning = true;
            receiveThread = new Thread(ReceiveLoop);
            receiveThread.IsBackground = true;
            receiveThread.Priority = System.Threading.ThreadPriority.AboveNormal;
            receiveThread.Start();

            Debug.Log($"[UDPManager] Initialized (low-latency mode) - Receiving on port {receivePort}, Sending to {targetIP}:{targetPort}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[UDPManager] Failed to initialize: {e.Message}");
        }
    }

    public void Shutdown()
    {
        isRunning = false;

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(1000);
        }

        receiveClient?.Close();

        Debug.Log("[UDPManager] Shutdown complete");
    }

    void Update()
    {
        if (!isRunning)
        {
            StartUDP();
            return;
        }

        // Process queued messages on main thread (limit per frame to prevent stuttering)
        ProcessMessageQueue(maxMessagesPerFrame: 50);

        // Clean up old chunked messages periodically
        CleanupOldChunkedMessages();
    }

    private void ReceiveLoop()
    {
        IPEndPoint anyEndpoint = new IPEndPoint(IPAddress.Any, 0);

        while (isRunning)
        {
            try
            {
                byte[] data = receiveClient.Receive(ref anyEndpoint);
                string json = Encoding.UTF8.GetString(data);

                lock (lockObject)
                {
                    // Drop old messages if queue is getting too large (prioritize freshness)
                    while (messageQueue.Count > 100)
                    {
                        messageQueue.Dequeue();
                    }
                    messageQueue.Enqueue(json);
                }
            }
            catch (SocketException e)
            {
                // Ignore timeout exceptions (expected with receive timeout)
                if (isRunning && e.SocketErrorCode != SocketError.TimedOut)
                {
                    Debug.LogWarning($"[UDPManager] Socket error: {e.Message}");
                }
            }
            catch (Exception e)
            {
                if (isRunning)
                {
                    Debug.LogError($"[UDPManager] Receive error: {e.Message}");
                }
            }
        }
    }

    private void ProcessMessageQueue(int maxMessagesPerFrame = 50)
    {
        int processed = 0;
        lock (lockObject)
        {
            while (messageQueue.Count > 0 && processed < maxMessagesPerFrame)
            {
                string json = messageQueue.Dequeue();
                ProcessMessage(json);
                processed++;
            }
        }
    }

    private void ProcessMessage(string json)
    {
        try
        {
            var wrapper = JsonUtility.FromJson<MessageWrapper>(json);

            if (debugMode)
            {
                Debug.Log($"[UDPManager] Received message type: {wrapper.type}");
            }

            // Handle chunked messages
            if (wrapper.type == "chunk")
            {
                var chunk = JsonUtility.FromJson<ChunkWrapper>(json);
                HandleChunkedMessage(chunk);
                return;
            }

            // Handle regular messages
            switch (wrapper.type)
            {
                case "joint_states":
                    var jointData = JsonUtility.FromJson<JointStateMessage>(json);
                    OnJointStateReceived?.Invoke(jointData.data);
                    break;

                case "color_image":
                    var colorData = JsonUtility.FromJson<ImageMessage>(json);
                    OnColorImageReceived?.Invoke(colorData.data);
                    break;

                case "depth_image":
                    var depthData = JsonUtility.FromJson<DepthImageMessage>(json);
                    OnDepthImageReceived?.Invoke(depthData.data);
                    break;

                case "camera_info":
                    var cameraInfoData = JsonUtility.FromJson<CameraInfoMessage>(json);
                    OnCameraInfoReceived?.Invoke(cameraInfoData.data);
                    break;

                default:
                    if (debugMode)
                    {
                        Debug.LogWarning($"[UDPManager] Unknown message type: {wrapper.type}");
                    }
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[UDPManager] Failed to parse message: {e.Message}");
        }
    }

    private void HandleChunkedMessage(ChunkWrapper chunk)
    {
        if (!chunkedMessages.TryGetValue(chunk.message_id, out ChunkedMessage msg))
        {
            msg = new ChunkedMessage(chunk.total_chunks, chunk.msg_type);
            chunkedMessages[chunk.message_id] = msg;
        }

        // Decode and store chunk
        byte[] chunkData = Convert.FromBase64String(chunk.data);
        msg.chunks[chunk.chunk_index] = chunkData;
        msg.receivedCount++;

        // Check if message is complete
        if (msg.receivedCount == msg.totalChunks)
        {
            // Reconstruct full message
            List<byte> fullData = new List<byte>();
            foreach (var c in msg.chunks)
            {
                fullData.AddRange(c);
            }

            string fullJson = Encoding.UTF8.GetString(fullData.ToArray());
            chunkedMessages.Remove(chunk.message_id);

            // Process reconstructed message
            ProcessMessage(fullJson);
        }
    }

    private void CleanupOldChunkedMessages()
    {
        float currentTime = Time.time;
        List<uint> toRemove = new List<uint>();

        foreach (var kvp in chunkedMessages)
        {
            if (currentTime - kvp.Value.timestamp > 5.0f) // 5 second timeout
            {
                toRemove.Add(kvp.Key);
            }
        }

        foreach (var key in toRemove)
        {
            chunkedMessages.Remove(key);
            if (debugMode)
            {
                Debug.LogWarning($"[UDPManager] Dropped incomplete chunked message: {key}");
            }
        }
    }


    public void SetTargetEndpoint(string ip, int port)
    {
        targetIP = ip;
        targetPort = port;
        remoteEndPoint = new IPEndPoint(IPAddress.Parse(targetIP), targetPort);
        Debug.Log($"[UDPManager] Target endpoint updated: {targetIP}:{targetPort}");
    }

    public string GetTargetIP()
    {
        return targetIP;
    }
}

// Wrapper for initial type detection
[Serializable]
public class MessageWrapper
{
    public string type;
}

// Chunk wrapper for large messages
[Serializable]
public class ChunkWrapper
{
    public string type;
    public string msg_type;
    public uint message_id;
    public int chunk_index;
    public int total_chunks;
    public string data;
}

// Joint State
[Serializable]
public class JointStateMessage
{
    public string type;
    public JointStateData data;
}

[Serializable]
public class JointStateData
{
    public string[] names;
    public float[] positions;
    public float[] velocities;
    public float[] efforts;
    public int timestamp_sec;
    public int timestamp_nanosec;
    public string frame_id;
}

// Image
[Serializable]
public class ImageMessage
{
    public string type;
    public ImageData data;
}

[Serializable]
public class ImageData
{
    public string format;
    public string data; // base64 encoded
    public int timestamp_sec;
    public int timestamp_nanosec;
    public string frame_id;
}

// Arm Trajectory (for sending)
[Serializable]
public class ArmTrajectoryMessage
{
    public string type;
    public ArmTrajectoryData data;
}

[Serializable]
public class ArmTrajectoryData
{
    public float[] left;
    public float[] right;
}

// Cmd Vel (for sending)
[Serializable]
public class CmdVelMessage
{
    public string type;
    public CmdVelData data;
}

[Serializable]
public class CmdVelData
{
    public Vector3Data linear;
    public Vector3Data angular;
}

[Serializable]
public class Vector3Data
{
    public float x;
    public float y;
    public float z;
}

// Target Height offset (for sending)
[Serializable]
public class TargetHeightMessage
{
    public string type;
    public TargetHeightData data;
}

[Serializable]
public class TargetHeightData
{
    public float diff; // m, offset from policy's default base height
}

// Record Bag (for sending)
[Serializable]
public class RecordBagMessage
{
    public string type;
    public RecordBagData data;
}

[Serializable]
public class RecordBagData
{
    public int command; // 1=start, 0=stop&save, -1=stop&discard
}

// Camera Info (from ROS, for depth-based point cloud reconstruction)
[Serializable]
public class CameraInfoMessage
{
    public string type;
    public CameraInfoData data;
}

[Serializable]
public class CameraInfoData
{
    public float fx;
    public float fy;
    public float cx;
    public float cy;
    public int width;
    public int height;
}

// Depth Image (16-bit PNG with dimensions)
[Serializable]
public class DepthImageMessage
{
    public string type;
    public DepthImageData data;
}

[Serializable]
public class DepthImageData
{
    public int width;
    public int height;
    public string data; // base64 encoded 16-bit PNG bytes
}

