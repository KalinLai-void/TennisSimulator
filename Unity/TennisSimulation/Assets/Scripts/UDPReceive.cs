using UnityEngine;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using UnityEngine.UI;

public class UDPReceive : MonoBehaviour
{
    IPEndPoint senderIP;
    UdpClient client;
    public int port = 5054;
    public bool startRecieving = true;
    public bool printToConsole = false;
    private string data;

    // for tennis setting
    public Text textComponent;
    public GameObject ballPrefab;

    private void Start()
    {
        client = new UdpClient(port);
        senderIP = new IPEndPoint(IPAddress.Any, 0);
    }

    private void Update()
    {
        ReceiveData();
    }

    private void ReceiveData()
    {
        if (client != null)
        {
            if (client.Available > 0)
            {
                try
                {
                    byte[] dataByte = client.Receive(ref senderIP);
                    data = Encoding.UTF8.GetString(dataByte);

                    if (printToConsole) print(data);

                    ProcessData();
                }
                catch (Exception err)
                {
                    print(err.ToString());
                }
            }
        }
    }

    private void ProcessData()
    {
        if (data.Length >= 6) // remove "[" and "]"
        {
            data = data.Substring(1, data.Length - 2);
        }
        string[] info = data.Split(',');
        if (info.Length >= 6)
        {
            float initialSpeed = float.Parse(info[2]);
            float airAngle = float.Parse(info[1]);
            float sideAngle = float.Parse(info[0]);
            float startX = float.Parse(info[5]);
            float startY = float.Parse(info[4]);
            float startZ = float.Parse(info[3]);
            textComponent.text = "Initial Speed: " + initialSpeed
                                + "\nAir Angle: " + airAngle
                                + "\nSide Angle: " + sideAngle;

            // generate tennis
            GameObject ballObject = Instantiate(ballPrefab, transform.position, transform.rotation);
            TennisMotion ballScript = ballObject.GetComponent<TennisMotion>();
            ballObject.transform.position = new Vector3(startX, startY, startZ);
            ballScript.SetInitialVelocity(initialSpeed, airAngle, sideAngle);
            data = ""; // clear data
        }
    }

}
