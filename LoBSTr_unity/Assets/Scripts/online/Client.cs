using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Valve.VR;
using AsyncIO;

public class Client : MonoBehaviour
{
    public enum Mode
    {
        Animation = 0,
        VR
    }
    public Mode mode;
    public int target_framerate;

    public string filename;
    float[,] data;
    public int length;
    public int frame_count;

    public bool calibrated;
    public GameObject Character;
    public GameObject Reference;
    public GameObject[] Trackers;
    public GameObject[] Lowerbody_Joints;
    public float height_coeff;
    public Vector3[] past_positions;
    public Quaternion[] past_rotations;

    public bool Lcontact, Rcontact;

    private RequestSocket requestSocket;
    public List<string> input_list;

    void Start()
    {
        ForceDotNet.Force();

        if (mode==Mode.Animation)
        {
            Load_Animation();
            length = data.GetLength(0);
            frame_count = 0;
            calibrated = true;    
        }

        Application.targetFrameRate = target_framerate;

        requestSocket = new RequestSocket();
        requestSocket.Connect("tcp://localhost:3550");
        
        input_list = new List<string>();

        height_coeff = 1f;
    }

    void FixedUpdate()
    {
        if(mode==Mode.Animation)
        {
            if (frame_count >= length)
                frame_count = 0;               

            PlayAnimation();
        }
        else
        {
            if (SteamVR_Actions.default_GrabGrip.GetStateUp(SteamVR_Input_Sources.Any))
                CalibrateTrackers();      
        }

        if(calibrated)
        {
            string input = GenerateInput();
            input_list.Add(input);

            if (input_list.Count > target_framerate)
                input_list.RemoveAt(0);

            if (input_list.Count == target_framerate)         
                PredictLowerPose();
        }    
        frame_count++;
    }

    private void OnApplicationQuit() {           
        requestSocket.Close();
        NetMQConfig.Cleanup();
    }

    void CalibrateTrackers() {
        if (!calibrated)
        {
            height_coeff = 1f / Trackers[0].transform.position.y;
            Reference.transform.localScale /= height_coeff;
            Character.transform.localScale /= height_coeff;

            foreach (GameObject t in Trackers)
                t.transform.GetComponent<SteamVR_TrackedObject>().Calibrate();
            
            Reference.GetComponent<ReferenceTransform>().enabled = true;
            calibrated = true;
        }  
    }
    string GenerateInput()
    {
        string input_string = "";
        Vector3 ref_w_pos = Reference.transform.position;
        Quaternion ref_w_rot = Reference.transform.rotation;
        float ref_height = ref_w_pos.y * height_coeff;

        for (int i = 0; i < 4; i++)
        {
            Vector3 joint_pos = Trackers[i].transform.position;
            Quaternion joint_rot = Trackers[i].transform.rotation;

            Vector3 joint_delta_p = Quaternion.Inverse(ref_w_rot) * (joint_pos - past_positions[i]);
            Quaternion joint_delta_q = Quaternion.Inverse(ref_w_rot) *(joint_rot * Quaternion.Inverse(past_rotations[i]));
            Matrix4x4 joint_delta_mat = Matrix4x4.Rotate(joint_delta_q);
            Vector3 joint_delta_up = new Vector3(joint_delta_mat.m01, joint_delta_mat.m11, joint_delta_mat.m21);
            Vector3 joint_delta_forward = new Vector3(joint_delta_mat.m02, joint_delta_mat.m12, joint_delta_mat.m22);

            input_string += joint_delta_up.x.ToString("F8") + ", " + joint_delta_forward.x.ToString("F8") + ", " + joint_delta_p.x.ToString("F8") + ", ";
            input_string += joint_delta_up.y.ToString("F8") + ", " + joint_delta_forward.y.ToString("F8") + ", " + joint_delta_p.y.ToString("F8") + ", ";
            input_string += joint_delta_up.z.ToString("F8") + ", " + joint_delta_forward.z.ToString("F8") + ", " + joint_delta_p.z.ToString("F8") + ", ";

            past_positions[i] = joint_pos;
            past_rotations[i] = joint_rot;
        }
        input_string += ref_height.ToString("F8");

        return input_string;
    }

    void PredictLowerPose() {
        requestSocket.SendFrame(string.Join(", ", input_list));
        string response = requestSocket.ReceiveFrameString();
        var splittedStrings = response.Split(' ');

        for (int i = 0; i < 8; i++)
        {
            Vector3 upward = new Vector3(float.Parse(splittedStrings[6 * i]), float.Parse(splittedStrings[6 * i + 2]), float.Parse(splittedStrings[6 * i + 4]));
            Vector3 forward = new Vector3(float.Parse(splittedStrings[6 * i + 1]), float.Parse(splittedStrings[6 * i + 3]), float.Parse(splittedStrings[6 * i + 5]));
            Lowerbody_Joints[i].transform.localRotation = Quaternion.LookRotation(forward, upward);
        }

        Lcontact = float.Parse(splittedStrings[48]) == 1f ? true : false;
        Rcontact = float.Parse(splittedStrings[49]) == 1f ? true : false;
    }

    void Load_Animation()
    {
        string path = "/Animations/" + filename + ".csv";
        var fileData = System.IO.File.ReadAllText(Application.dataPath + path);

        var lines = fileData.Split("\n"[0]);
        int row_count = lines.Length;
        var lineData = (lines[0].Trim()).Split(","[0]);
        int column_count = lineData.Length;
        var x = 0f;
        float.TryParse(lineData[0], out x);

        data = new float[row_count, column_count];

        for (int i = 0; i < row_count - 1; i++)
        {
            var Line = (lines[i].Trim()).Split(","[0]);
            for (int j = 0; j < column_count; j++)
            {
                var v = 0f;
                float.TryParse(Line[j], out v);
                data[i, j] = v;
            }
        }
    }

    void PlayAnimation() {
        for (int j = 0; j < Trackers.Length; j++)
        {
            Vector3 up = new Vector3(data[frame_count, 9 * j]
                , data[frame_count, 9 * j + 3]
                , data[frame_count, 9 * j + 6]);
            Vector3 forward = new Vector3(data[frame_count, 9 * j + 1]
                , data[frame_count, 9 * j + 4]
                , data[frame_count, 9 * j + 7]);
            Vector3 pos = new Vector3(data[frame_count, 9 * j + 2]
                , data[frame_count, 9 * j + 5]
                , data[frame_count, 9 * j + 8]);
            
            Trackers[j].transform.localPosition = pos;
            Trackers[j].transform.localRotation = Quaternion.LookRotation(forward, up);
        }

        if(!calibrated){
            for (int j = 0; j < Lowerbody_Joints.Length; j++)
            {
                Vector3 up = new Vector3(data[frame_count, 36 + 6*j]
                    , data[frame_count, 36 + 6*j + 2]
                    , data[frame_count, 36 + 6*j + 4]);
                Vector3 forward = new Vector3(data[frame_count, 36 + 6*j + 1]
                    , data[frame_count, 36 + 6*j + 3]
                    , data[frame_count, 36 + 6*j + 5]);
                Lowerbody_Joints[j].transform.localRotation = Quaternion.LookRotation(forward, up);
            }          
        }
    }
}
