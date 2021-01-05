using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(PlayCSV))]
public class PlacCSV_Editor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        PlayCSV playcsv = (PlayCSV)target;

        if (GUILayout.Button("Load Animation")) {
            playcsv.Load_Animation(playcsv.file_path);
        }

        if (GUILayout.Button("Load Joints"))
        {
            playcsv.Load_Joints();
        }

        if (GUILayout.Button("Play Animation")) {
            playcsv.isPlay = true;
        }

        if (GUILayout.Button("Pause Animation")) {
            playcsv.isPlay = false;
        }

        if (GUILayout.Button("Stop Animation"))
        {
            playcsv.current_frame = 0;
            playcsv.isPlay = false;
        }
    }
}
