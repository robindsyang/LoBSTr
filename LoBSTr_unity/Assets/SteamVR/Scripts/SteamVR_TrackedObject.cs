//======= Copyright (c) Valve Corporation, All rights reserved. ===============
//
// Purpose: For controlling in-game objects with tracked devices.
//
//=============================================================================
using System.Text;
using System.Collections.Generic;
using UnityEngine;
using Valve.VR;

namespace Valve.VR
{
    public class SteamVR_TrackedObject : MonoBehaviour
    {
        public enum EIndex
        {
            None = -1,
            Hmd = (int)OpenVR.k_unTrackedDeviceIndex_Hmd,
            Device1,
            Device2,
            Device3,
            Device4,
            Device5,
            Device6,
            Device7,
            Device8,
            Device9,
            Device10,
            Device11,
            Device12,
            Device13,
            Device14,
            Device15
        }
        public int FixedIdx;
        public EIndex index;

        [Tooltip("If not set, relative to parent")]
        public Transform origin;

        [Tooltip("Offsets")]
        public Vector3 localPosOffset;
        public Vector3 localRotOffset;

        [Tooltip("Smoothing")]
        public bool denoisingOn = false;
        public List<Vector3> TrackedPosQueue;
        public List<Vector3> TrendQueue;
        public int windowSize = 3;
        public float k_uncertainty = 2.17f; // 1% = 2.576, 2% = 2.325, 3% = 2.17, 4% = 2.055, 5% = 1.960
        public float jitterRadius = 0.01f; // 1cm
        public float expSmoothingParam = 0.5f;
        public float correctionParam = 0.1f;

        public bool isValid { get; private set; }

        public GameObject CalibTarget;

        private void OnNewPoses(TrackedDevicePose_t[] poses)
        {
            if (index == EIndex.None)
                return;

            var i = (int)index;

            isValid = false;
            if (poses.Length <= i)
                return;

            if (!poses[i].bDeviceIsConnected)
                return;

            if (!poses[i].bPoseIsValid)
                return;

            isValid = true;

            var pose = new SteamVR_Utils.RigidTransform(poses[i].mDeviceToAbsoluteTracking);

            // add offset values
            if (origin != null)
            {
                transform.rotation = origin.rotation * pose.rot * Quaternion.Euler(localRotOffset);
                transform.position = origin.transform.TransformPoint(pose.pos) + transform.TransformVector(localPosOffset);
            }
            else
            {
                transform.localRotation = pose.rot * Quaternion.Euler(localRotOffset);
                transform.localPosition = pose.pos + transform.TransformVector(localPosOffset);
            }

            // Smoothing
            // moving average filter
            if (denoisingOn)
            {
                transform.position = MovingAverageFilter();
            }
        }

        SteamVR_Events.Action newPosesAction;

        SteamVR_TrackedObject()
        {
            newPosesAction = SteamVR_Events.NewPosesAction(OnNewPoses);
        }

        private void Awake()
        {
            OnEnable();
            AssignDeviceIndex();
        }

        void OnEnable()
        {
            var render = SteamVR_Render.instance;
            if (render == null)
            {
                enabled = false;
                return;
            }

            newPosesAction.enabled = true;
        }

        void OnDisable()
        {
            newPosesAction.enabled = false;
            isValid = false;
        }

        string GetSerialNum(int idx)
        {
            uint index = (uint)idx;
            ETrackedPropertyError error = new ETrackedPropertyError();
            StringBuilder sb = new StringBuilder();
            OpenVR.System.GetStringTrackedDeviceProperty(index, ETrackedDeviceProperty.Prop_SerialNumber_String, sb, OpenVR.k_unMaxPropertyStringSize, ref error);
            string probablyUniqueDeviceSerial = sb.ToString();
            return probablyUniqueDeviceSerial;
        }

        public void SetDeviceIndex(int index)
        {
            if (System.Enum.IsDefined(typeof(EIndex), index))
                this.index = (EIndex)index;
        }

        void AssignDeviceIndex() {
            switch (FixedIdx) {
                case 1: // left hand
                    for (int i = 2; i < 16; i++)
                    {
                        if (GetSerialNum(i) == "LHR-49EF6B64" || GetSerialNum(i) == "LHR-FF537947" || GetSerialNum(i) == "LHR-FF3F1942" || GetSerialNum(i) == "LHR-EA380899" || GetSerialNum(i) == "LHR-04DEC87A")
                        {
                            SetDeviceIndex(i);
                        }
                    }
                    break;
                case 2: // right hand
                    for (int i = 2; i < 16; i++)
                    {
                        if (GetSerialNum(i) == "LHR-CD0A7277" || GetSerialNum(i) == "LHR-F7CDB947" || GetSerialNum(i) == "LHR-F747DB42" || GetSerialNum(i) == "LHR-A3E69315" || GetSerialNum(i) == "LHR-6723A758")
                        {
                            SetDeviceIndex(i);
                        }
                    }
                    break;
                case 3: // pelvis
                    for (int i = 2; i < 16; i++) {
                        if (GetSerialNum(i) == "LHR-0DC0C48B" || GetSerialNum(i) == "LHR-05BAA719" || GetSerialNum(i) == "LHR-078F5A82" || GetSerialNum(i) == "LHR-B66DD003") {
                            SetDeviceIndex(i);
                        }
                    }
                    break;
                case 4: // lfoot
                    for (int i = 2; i < 16; i++)
                    {
                        if (GetSerialNum(i) == "LHR-1FDC1022" || GetSerialNum(i) == "LHR-06BFAF29" || GetSerialNum(i) == "LHR-C78FD5D4" || GetSerialNum(i) == "LHR-8B1D59B7" || GetSerialNum(i) == "LHR-4C3619D8")
                        {
                            SetDeviceIndex(i);
                        }
                    }
                    break;
                case 5: // rfoot
                    for (int i = 2; i < 16; i++)
                    {
                        if (GetSerialNum(i) == "LHR-1BDE601B" || GetSerialNum(i) == "LHR-14BF03F2" || GetSerialNum(i) == "LHR-5BA9B75D" || GetSerialNum(i) == "LHR-E1C160BE" || GetSerialNum(i) == "LHR-B0DE4E66")
                        {
                            SetDeviceIndex(i);
                        }
                    }
                    break;
            }
        }

        public void Calibrate()
        {
            if (FixedIdx != 3)
            {
                localRotOffset = (Quaternion.Inverse(transform.rotation) * CalibTarget.transform.rotation).eulerAngles;
            }
            //localPosOffset = transform.InverseTransformDirection(CalibTarget.transform.position - transform.position);          
        }

        Vector3 MovingAverageFilter()
        {
            Vector3 trendVec = Vector3.zero;
            Vector3 rawPos = transform.position;
            Vector3 filteredPos = rawPos;

            if (TrackedPosQueue.Count == 3)
            {
                //jitter filter
                float dist = Vector3.Distance(TrackedPosQueue[windowSize - 1], rawPos);


                if (dist <= jitterRadius)
                    filteredPos = rawPos * (dist / jitterRadius) + TrackedPosQueue[windowSize - 1] * (1f - dist / jitterRadius); // ?
                else
                    filteredPos = rawPos;

                filteredPos = filteredPos * (1f - expSmoothingParam) + expSmoothingParam * (TrackedPosQueue[windowSize - 1] + TrendQueue[windowSize - 1]);
                Vector3 diffVec = filteredPos - TrackedPosQueue[windowSize - 1];
                trendVec = diffVec * correctionParam + TrendQueue[windowSize - 1] * (1f - correctionParam);

                TrackedPosQueue.RemoveAt(0);
                TrackedPosQueue.Add(filteredPos);
                TrendQueue.RemoveAt(0);
                TrendQueue.Add(trendVec);
            }
            return filteredPos;
        }
    }
}