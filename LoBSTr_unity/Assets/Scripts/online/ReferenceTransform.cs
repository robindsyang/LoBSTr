using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ReferenceTransform : MonoBehaviour
{
    public GameObject Pelvis;

    // Update is called once per frame
    void Update()
    {
        transform.position = Pelvis.transform.position;
        Vector3 forward = new Vector3(Pelvis.transform.forward.x, 0f, Pelvis.transform.forward.z);
        transform.rotation = Quaternion.LookRotation(forward, Vector3.up);
    }
}
