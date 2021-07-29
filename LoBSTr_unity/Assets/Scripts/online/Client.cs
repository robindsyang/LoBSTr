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
    Mode mode;
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

    public bool IK_postprocessing;
    public GameObject[] LLegJoints;
    public GameObject[] RLegJoints;
    public GameObject[] Feet;
    public bool L_prev_contact;
    public bool R_prev_contact;
    public Vector3 L_fixed_position;
    public Vector3 R_fixed_position;
    public float[,] L_Jacobian;
    public float[,] R_Jacobian;
    public int iter;
    public int blending_window;
    public float L_blend_parameter;
    public float R_blend_parameter;
    float l_p, r_p;

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

        L_prev_contact = false;
        R_prev_contact = false;
        L_fixed_position = Vector3.zero;
        R_fixed_position = Vector3.zero;

        height_coeff = 1f;
        L_blend_parameter = 1;
        R_blend_parameter = 1;
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

        bool contact_left = float.Parse(splittedStrings[48]) == 1f ? true : false;
        bool contact_right = float.Parse(splittedStrings[49]) == 1f ? true : false;

        // IK post processing
        if (IK_postprocessing)
        {
            if (contact_left)
            {
                if (!L_prev_contact)
                {
                    L_fixed_position = Feet[0].transform.position;
                    L_fixed_position.y = 0f;
                }

                IK_position(L_fixed_position, Feet[0], LLegJoints);
                L_prev_contact = true;
                L_blend_parameter = 0;
            }
            else
            {
                l_p = -Mathf.Sin(L_blend_parameter * Mathf.PI / (float)blending_window + Mathf.PI / 2f) + 1f;

                Vector3 current_foot_pos = Feet[0].transform.position;
                Vector3 target_foot_pos = Vector3.Lerp(L_fixed_position, current_foot_pos, l_p);
                IK_position(target_foot_pos, Feet[0], LLegJoints);

                if (L_blend_parameter < blending_window)
                    L_blend_parameter++;
                else
                    L_blend_parameter = blending_window;

                L_prev_contact = false;
            }

            if (contact_right)
            {
                if (!R_prev_contact)
                {
                    R_fixed_position = Feet[1].transform.position;
                    R_fixed_position.y = 0f;
                }

                IK_position(R_fixed_position, Feet[1], RLegJoints);
                R_prev_contact = true;
                R_blend_parameter = 0;
            }
            else
            {
                r_p = -Mathf.Sin(R_blend_parameter * Mathf.PI / (float)blending_window + Mathf.PI / 2f) + 1f;

                Vector3 current_foot_pos = Feet[1].transform.position;
                Vector3 target_foot_pos = Vector3.Lerp(R_fixed_position, current_foot_pos, r_p);
                IK_position(target_foot_pos, Feet[1], RLegJoints);

                if (R_blend_parameter < blending_window)
                    R_blend_parameter++;
                else
                    R_blend_parameter = blending_window;

                R_prev_contact = false;
            }
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

    void IK_position(Vector3 target_p, GameObject EE, GameObject[] Joints)
    {
        int n = 0;
        while (n < iter)
        {
            Vector3 target_v = EE.transform.InverseTransformVector(target_p - EE.transform.position);

            if (target_v.magnitude < 0.01f)
                break;

            double[] d_target_v = new double[3];
            d_target_v[0] = target_v.x;
            d_target_v[1] = target_v.y;
            d_target_v[2] = target_v.z;
            Vector<double> v_d_target_v = DenseVector.OfArray(d_target_v);

            double[,] J = UpdateLeftLegtBodyJacobian(EE, Joints);
            Matrix<double> m_J_linear = DenseMatrix.OfArray(J);
            Vector<double> d_theta = v_d_target_v * m_J_linear;

            double[] delta_theta = d_theta.ToArray();

            for (int i = 0; i < delta_theta.Length / 3; i++)
            {
                Vector3 delta_euler = new Vector3((float)delta_theta[3 * i], (float)delta_theta[3 * i + 1], (float)delta_theta[3 * i + 2]);
                Vector3 joint_euler = Joints[i].transform.localRotation.eulerAngles;
                Joints[i].transform.localRotation = Quaternion.Euler(joint_euler + delta_euler);
            }
            n++;
        }
    }

    void Adjoint_T(Vector3 p, Quaternion R, Vector3 w_in, Vector3 v_in, out Vector3 w_out, out Vector3 v_out)
    {
        w_out = R * w_in;
        v_out = Vector3.Cross(p, R * w_in) + R * v_in;
    }

    double[,] UpdateLeftLegtBodyJacobian(GameObject EE, GameObject[] Joints)
    {
        double[,] Jacobian = new double[3, 12];
        Vector3[] Jw = new Vector3[12]; // top three rows (for rotation) of Jacobian
        Vector3[] Jv = new Vector3[12]; // bottom three rows (for translation) of Jacobian

        Adjoint_T(EE.transform.InverseTransformPoint(Joints[0].transform.position), Quaternion.Inverse(EE.transform.rotation) * Joints[0].transform.rotation, new Vector3(1f, 0f, 0f), new Vector3(0f, 0f, 0f), out Jw[0], out Jv[0]);
        Adjoint_T(EE.transform.InverseTransformPoint(Joints[0].transform.position), Quaternion.Inverse(EE.transform.rotation) * Joints[0].transform.rotation, new Vector3(0f, 1f, 0f), new Vector3(0f, 0f, 0f), out Jw[1], out Jv[1]);
        Adjoint_T(EE.transform.InverseTransformPoint(Joints[0].transform.position), Quaternion.Inverse(EE.transform.rotation) * Joints[0].transform.rotation, new Vector3(0f, 0f, 1f), new Vector3(0f, 0f, 0f), out Jw[2], out Jv[2]);

        Adjoint_T(EE.transform.InverseTransformPoint(Joints[1].transform.position), Quaternion.Inverse(EE.transform.rotation) * Joints[1].transform.rotation, new Vector3(1f, 0f, 0f), new Vector3(0f, 0f, 0f), out Jw[3], out Jv[3]);
        Adjoint_T(EE.transform.InverseTransformPoint(Joints[1].transform.position), Quaternion.Inverse(EE.transform.rotation) * Joints[1].transform.rotation, new Vector3(0f, 1f, 0f), new Vector3(0f, 0f, 0f), out Jw[4], out Jv[4]);
        Adjoint_T(EE.transform.InverseTransformPoint(Joints[1].transform.position), Quaternion.Inverse(EE.transform.rotation) * Joints[1].transform.rotation, new Vector3(0f, 0f, 1f), new Vector3(0f, 0f, 0f), out Jw[5], out Jv[5]);

        Adjoint_T(EE.transform.InverseTransformPoint(Joints[2].transform.position), Quaternion.Inverse(EE.transform.rotation) * Joints[2].transform.rotation, new Vector3(1f, 0f, 0f), new Vector3(0f, 0f, 0f), out Jw[6], out Jv[6]);
        Adjoint_T(EE.transform.InverseTransformPoint(Joints[2].transform.position), Quaternion.Inverse(EE.transform.rotation) * Joints[2].transform.rotation, new Vector3(0f, 1f, 0f), new Vector3(0f, 0f, 0f), out Jw[7], out Jv[7]);
        Adjoint_T(EE.transform.InverseTransformPoint(Joints[2].transform.position), Quaternion.Inverse(EE.transform.rotation) * Joints[2].transform.rotation, new Vector3(0f, 0f, 1f), new Vector3(0f, 0f, 0f), out Jw[8], out Jv[8]);

        Adjoint_T(EE.transform.InverseTransformPoint(Joints[3].transform.position), Quaternion.Inverse(EE.transform.rotation) * Joints[3].transform.rotation, new Vector3(1f, 0f, 0f), new Vector3(0f, 0f, 0f), out Jw[9], out Jv[9]);
        Adjoint_T(EE.transform.InverseTransformPoint(Joints[3].transform.position), Quaternion.Inverse(EE.transform.rotation) * Joints[3].transform.rotation, new Vector3(0f, 1f, 0f), new Vector3(0f, 0f, 0f), out Jw[10], out Jv[10]);
        Adjoint_T(EE.transform.InverseTransformPoint(Joints[3].transform.position), Quaternion.Inverse(EE.transform.rotation) * Joints[3].transform.rotation, new Vector3(0f, 0f, 1f), new Vector3(0f, 0f, 0f), out Jw[11], out Jv[11]);

        // Pack Jw[] and Jv[] into J_lleg
        for (int i = 0; i < 12; ++i)
        {
            // we only use Jv for here (IK solved for position only)
            Jacobian[0, i] = Jv[i].x;
            Jacobian[1, i] = Jv[i].y;
            Jacobian[2, i] = Jv[i].z;
        }

        return Jacobian;
    }
}
