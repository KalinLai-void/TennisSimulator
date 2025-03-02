using UnityEngine;
using UnityEngine.UI;

public class TennisMotion : MonoBehaviour
{
    private float mass; // 網球質量
    private float dragCoefficient; // 網球阻力係數
    private const float magnusCoefficient = 0.2f; // 網球馬格努斯力係數
    private float ballRadius;
    private float gravity;
    public float airDensity = 1.205f; // 空氣密度 (20度C = 1.205)

    private bool hasTriggered = false;
    public GameObject pointPrefab;

    // Start is called before the first frame update
    void Awake()
    {
        Rigidbody ballRigidbody = GetComponent<Rigidbody>();
        SphereCollider ballCollider = GetComponent<SphereCollider>();
        if (ballRigidbody && ballCollider)
        {
            mass = ballRigidbody.mass;
            dragCoefficient = ballRigidbody.drag;
            ballRadius = ballCollider.radius;
            gravity = Physics.gravity.y;
        }
    }

    public void SetInitialVelocity(float initialSpeed, float airAngle, float sideAngle)
    {
        // get vertical angle and horizontal angle (radian)
        float radianXY = Mathf.Deg2Rad * sideAngle;
        float radianXZ = Mathf.Deg2Rad * airAngle;

        // get fly time
        // float t = (2f * initialSpeed * Mathf.Sin(sideAngle * Mathf.Deg2Rad)) / g;

        // get the X component and Z component of the velocity
        float speedZ = initialSpeed * Mathf.Cos(radianXY) * Mathf.Cos(radianXZ);
        float speedY = initialSpeed * Mathf.Sin(radianXY);
        float speedX = initialSpeed * Mathf.Cos(radianXY) * Mathf.Sin(radianXZ);
        Vector3 initialVelocity = new Vector3(speedX, speedY, speedZ);

        Rigidbody ballRigidbody = GetComponent<Rigidbody>();
        ballRigidbody.velocity = initialVelocity;

        Vector3 airResistance = CalculateAirResistance(initialVelocity);
        Vector3 magnusForce = CalculateMagnusForce(initialVelocity);
        Vector3 gravityForce = new Vector3(0, mass * gravity, 0);

        Vector3 totalForce = airResistance + magnusForce + gravityForce;
        Vector3 acceleration = totalForce / mass;
        
        GetComponent<Rigidbody>().AddForce(acceleration, ForceMode.Acceleration);

    }

    private Vector3 CalculateAirResistance(Vector3 initialVelocity)
    {
        float area = Mathf.PI * ballRadius * ballRadius;
        Vector3 direction = -1 * GetComponent<Rigidbody>().velocity.normalized;
        float airResistance = 0.5f * airDensity * dragCoefficient * area * initialVelocity.magnitude * initialVelocity.magnitude;
        return direction * airResistance;
    }

    private Vector3 CalculateMagnusForce(Vector3 initialVelocity)
    {
        Vector3 direction = Vector3.Cross(GetComponent<Rigidbody>().angularVelocity, initialVelocity);
        float area = Mathf.PI * ballRadius * ballRadius;
        return magnusCoefficient * area * direction * airDensity;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision != null)
        {
            if (!hasTriggered)
            {
                ContactPoint contact = collision.contacts[0];
                Quaternion rotation = Quaternion.FromToRotation(Vector3.up, contact.normal);
                Vector3 position = contact.point;

                CourtInOut court = collision.gameObject.GetComponent<CourtInOut>();
                if (court)
                {
                    Instantiate(pointPrefab, position, rotation);

                    if (position.x >= 0 && position.x <= 3.31294
                        && position.z >= 3.59078 && position.z <= 7.18156)
                    {
                        Debug.Log("Position: " + position + " IN");
                        GameObject.Find("ResultText").GetComponent<Text>().text = "IN";
                    }
                    else
                    {
                        Debug.Log("Position: " + position + " OUT");
                        GameObject.Find("ResultText").GetComponent<Text>().text = "OUT";
                    }
                }
                else
                {
                    Debug.Log("Position: " + position + " OUT");
                    GameObject.Find("ResultText").GetComponent<Text>().text = "OUT";
                }
             
                hasTriggered = true;
                Destroy(gameObject, 3f);
            }
        }
    }

    private void OnDestroy()
    {
        GameObject.Find("ResultText").GetComponent<Text>().text = "";
    }
}
