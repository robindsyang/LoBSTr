using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

public class PlayCSV : MonoBehaviour
{
    public enum Representation {
        Local,
        World,
        Reflocal
    };

    public string file_path;
    public Representation representation;
    float[,] Animation;
    public int Length;

    public GameObject Root;
    public List<GameObject> Joints;

    public bool isPlay;
    public int current_frame = 0;

    private void Update()
    {
        if (isPlay) {
            for (int j=0; j < Joints.Count; j++)
            {
                Vector3 up = new Vector3(Animation[current_frame, 9 * j]
                          , Animation[current_frame, 9 * j + 3]
                          , Animation[current_frame, 9 * j + 6]);
                Vector3 forward = new Vector3(Animation[current_frame, 9 * j + 1]
                        , Animation[current_frame, 9 * j + 4]
                        , Animation[current_frame, 9 * j + 7]);
                Vector3 pos = new Vector3(Animation[current_frame, 9 * j + 2]
                        , Animation[current_frame, 9 * j + 5]
                        , Animation[current_frame, 9 * j + 8]);

                if (representation==Representation.World)
                {
                    Joints[j].transform.position = pos;
                    Joints[j].transform.rotation = Quaternion.LookRotation(forward, up);
                }
                else
                {
                    Joints[j].transform.localPosition = pos;
                    Joints[j].transform.localRotation = Quaternion.LookRotation(forward, up);
                }
            }

            if (current_frame == Animation.GetLength(0) - 1)
                current_frame = 0;
            else
                current_frame++;
        }
    }

    public void Load_Joints()
    {
        if (Animation == null)
            Load_Animation(file_path);

        Joints.Clear();
        foreach (Transform child in Root.GetComponentsInChildren<Transform>())
        {
            if (child.name != "Sphere" && child.name != "Cylinder")
                Joints.Add(child.gameObject);
        }

        if (Animation.GetLength(1) / 9 != Joints.Count)
            Debug.LogError("Joint count not match!");
    }    

    public void Load_Animation(string path)
    {
        switch (representation) {
            case Representation.Local:
                path += "_local";
                break;
            case Representation.World:
                path += "_world";
                break;
            case Representation.Reflocal:
                path += "_reflocal";
                break;
        }

        var Data = System.IO.File.ReadAllText(Application.dataPath + path + ".csv");

        var lines = Data.Split("\n"[0]);
        int row_count = lines.Length;
        var lineData = (lines[0].Trim()).Split(","[0]);
        int column_count = lineData.Length;
        var x = 0f;
        float.TryParse(lineData[0], out x);

        Animation = new float[row_count, column_count];
        Length = row_count;

        for (int i = 0; i < row_count - 1; i++)
        {
            var Line = (lines[i].Trim()).Split(","[0]);
            for (int j = 0; j < column_count; j++)
            {
                var v = 0f;
                float.TryParse(Line[j], out v);
                Animation[i, j] = v;
            }
        }
    }
}
