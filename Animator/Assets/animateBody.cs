using UnityEngine;
using System.Collections;
using UnityEngine.UI;
using System.IO;
using System.Globalization;

public class animateBody : MonoBehaviour
{
//Create body
    private Vector3 head;
    private Vector3 neck;
    private Vector3 abdomen;
    private Vector3 torso;
    private Vector3 chest;
    private Vector3 leftShoulder;
    private Vector3 rightShoulder;
    private Vector3 leftElbow;
    private Vector3 rightElbow;
    private Vector3 leftArm;
    private Vector3 rightArm;
    private Vector3 leftHand;
    private Vector3 rightHand;
    private Vector3 hip;
    private Vector3 leftHip;
    private Vector3 rightHip;

    private GameObject headNeck;
    private GameObject neckTorso;
    private GameObject torsoLShoulder;
    private GameObject torsoRShoulder;
    private GameObject LShoulderElbow;
    private GameObject LElbowArm;
    private GameObject LArmHand;
    private GameObject RShoulderElbow;
    private GameObject RElbowArm;
    private GameObject RArmHand;
    private GameObject torsoChest;
    private GameObject chestAbdomen;
    private GameObject abdomenHip;
    private GameObject hipLHip;
    private GameObject hipRHip;

    private GameObject RShoulder;
    private GameObject LShoulder;
    private GameObject Chest;
    private GameObject Torso;
    private GameObject Abdomen;
    private GameObject Hip;
    private GameObject LHip;
    private GameObject RHip;
    private GameObject RElbow;
    private GameObject LElbow;
    private GameObject RHand;
    private GameObject LHand;

   
    private Vector3 hipOld;
    private Vector3 leftHipOld;
    private Vector3 rightHipOld;
    private Vector3 headOld;
    private Vector3 neckOld;
    private Vector3 abdomenOld;
    private Vector3 torsoOld;
    private Vector3 chestOld;
    private Vector3 leftShoulderOld;
    private Vector3 rightShoulderOld;
    private Vector3 leftElbowOld;
    private Vector3 rightElbowOld;
    private Vector3 leftArmOld;
    private Vector3 rightArmOld;
    private Vector3 leftHandOld;
    private Vector3 rightHandOld;

    private Vector3 hipNew;
    private Vector3 leftHipNew;
    private Vector3 rightHipNew;
    private Vector3 headNew;
    private Vector3 neckNew;
    private Vector3 abdomenNew;
    private Vector3 torsoNew;
    private Vector3 chestNew;
    private Vector3 leftShoulderNew;
    private Vector3 rightShoulderNew;
    private Vector3 leftElbowNew;
    private Vector3 rightElbowNew;
    private Vector3 leftArmNew;
    private Vector3 rightArmNew;
    private Vector3 leftHandNew;
    private Vector3 rightHandNew;

    public TextAsset csvFile;
    public float xhipPos;
    public float yhipPos;
    public float zhipPos;
    public string[] records;
    public int frame;


    /**
    struct joint {
        public new Vector3 coordinates;
        public Quaternion rotations;

    public joint(new Vector3 coordinates, Quaternion rotations)
    {
        coordinates = coordinates;
        rotations = rotations;

    }
    }
    **/
    // Update is called once per frame

    void Start()
    {
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = 100;
        frame = 1;

        records = csvFile.text.Split('\n');
        string[] fields = records[0].Split(';');

        print(fields.Length);

        var culture = (CultureInfo)CultureInfo.CurrentCulture.Clone();
        headNeck = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        headNeck.name = "headNeck";
        neckTorso = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        neckTorso.name = "neckTorso";
        torsoLShoulder = GameObject.CreatePrimitive(PrimitiveType.Cube);
        torsoLShoulder.name = "torsoLSHoulder";
        torsoRShoulder = GameObject.CreatePrimitive(PrimitiveType.Cube);
        LShoulderElbow = GameObject.CreatePrimitive(PrimitiveType.Cube);
        LElbowArm = GameObject.CreatePrimitive(PrimitiveType.Cube);
        LShoulderElbow.name = "LShoulderElbow";
        LArmHand = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        LArmHand.name = "LArmHand";
        RShoulderElbow = GameObject.CreatePrimitive(PrimitiveType.Cube);
        RArmHand = GameObject.CreatePrimitive(PrimitiveType.Capsule); 
        RElbowArm = GameObject.CreatePrimitive(PrimitiveType.Cube); 
        torsoChest = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        chestAbdomen = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        abdomenHip = GameObject.CreatePrimitive(PrimitiveType.Cylinder);

        RShoulder = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        LShoulder = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        RElbow = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        LElbow = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        RHand = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        LHand = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        Chest = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        Torso = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        Abdomen = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        Hip = GameObject.CreatePrimitive(PrimitiveType.Sphere);
		
		// Distinguish between original ground truth files and....
        if (fields.Length == 154)
        {
            fields = records[1].Split(';');
            headOld = new Vector3(float.Parse(fields[35], culture) - xhipPos, float.Parse(fields[36], culture) - yhipPos, float.Parse(fields[37], culture) - zhipPos);
            neckOld = new Vector3(float.Parse(fields[28], culture) - xhipPos, float.Parse(fields[29], culture) - yhipPos, float.Parse(fields[30], culture) - zhipPos);
            abdomenOld = new Vector3(float.Parse(fields[7], culture) - xhipPos, float.Parse(fields[8], culture) - yhipPos, float.Parse(fields[9], culture) - zhipPos);
            chestOld = new Vector3(float.Parse(fields[14], culture) - xhipPos, float.Parse(fields[15], culture) - yhipPos, float.Parse(fields[16], culture) - zhipPos);
            torsoOld = new Vector3(float.Parse(fields[21], culture) - xhipPos, float.Parse(fields[22], culture) - yhipPos, float.Parse(fields[23], culture) - zhipPos);
            leftShoulderOld = new Vector3(float.Parse(fields[42], culture) - xhipPos, float.Parse(fields[43], culture) - yhipPos, float.Parse(fields[44], culture) - zhipPos);
            rightShoulderOld = new Vector3(float.Parse(fields[70], culture) - xhipPos, float.Parse(fields[71], culture) - yhipPos, float.Parse(fields[72], culture) - zhipPos);
            leftElbowOld = new Vector3(float.Parse(fields[56], culture) - xhipPos, float.Parse(fields[57], culture) - yhipPos, float.Parse(fields[58], culture) - zhipPos);
            rightElbowOld = new Vector3(float.Parse(fields[77], culture) - xhipPos, float.Parse(fields[78], culture) - yhipPos, float.Parse(fields[79], culture) - zhipPos);
            leftArmOld = new Vector3(float.Parse(fields[56], culture) - xhipPos, float.Parse(fields[57], culture) - yhipPos, float.Parse(fields[58], culture) - zhipPos);
            rightArmOld = new Vector3(float.Parse(fields[84], culture) - xhipPos, float.Parse(fields[85], culture) - yhipPos, float.Parse(fields[86], culture) - zhipPos);
            leftHandOld = new Vector3(float.Parse(fields[63], culture) - xhipPos, float.Parse(fields[64], culture) - yhipPos, float.Parse(fields[65], culture) - zhipPos);
            rightHandOld = new Vector3(float.Parse(fields[91], culture) - xhipPos, float.Parse(fields[92], culture) - yhipPos, float.Parse(fields[93], culture) - zhipPos);
            hipOld = new Vector3(float.Parse(fields[0], culture) - xhipPos, float.Parse(fields[1], culture) - yhipPos, float.Parse(fields[2], culture) - zhipPos);
        }
		// Generated files (legs have been cleaned out from data)
        else if (fields.Length == 42)
        {
            headOld = new Vector3(float.Parse(fields[15], culture) - xhipPos, float.Parse(fields[16], culture) - yhipPos, float.Parse(fields[17], culture) - zhipPos);
            neckOld = new Vector3(float.Parse(fields[12], culture) - xhipPos, float.Parse(fields[13], culture) - yhipPos, float.Parse(fields[14], culture) - zhipPos);
            abdomenOld = new Vector3(float.Parse(fields[3], culture) - xhipPos, float.Parse(fields[4], culture) - yhipPos, float.Parse(fields[5], culture) - zhipPos);
            chestOld = new Vector3(float.Parse(fields[6], culture) - xhipPos, float.Parse(fields[7], culture) - yhipPos, float.Parse(fields[8], culture) - zhipPos);
            torsoOld = new Vector3(float.Parse(fields[9], culture) - xhipPos, float.Parse(fields[10], culture) - yhipPos, float.Parse(fields[11], culture) - zhipPos);
            leftShoulderOld = new Vector3(float.Parse(fields[18], culture) - xhipPos, float.Parse(fields[19], culture) - yhipPos, float.Parse(fields[20], culture) - zhipPos);
            rightShoulderOld = new Vector3(float.Parse(fields[30], culture) - xhipPos, float.Parse(fields[31], culture) - yhipPos, float.Parse(fields[32], culture) - zhipPos);
            leftElbowOld = new Vector3(float.Parse(fields[21], culture) - xhipPos, float.Parse(fields[22], culture) - yhipPos, float.Parse(fields[23], culture) - zhipPos);
            leftArmOld = new Vector3(float.Parse(fields[24], culture) - xhipPos, float.Parse(fields[25], culture) - yhipPos, float.Parse(fields[26], culture) - zhipPos);
            rightElbowOld = new Vector3(float.Parse(fields[33], culture) - xhipPos, float.Parse(fields[34], culture) - yhipPos, float.Parse(fields[35], culture) - zhipPos);
            leftHandOld = new Vector3(float.Parse(fields[27], culture) - xhipPos, float.Parse(fields[28], culture) - yhipPos, float.Parse(fields[29], culture) - zhipPos);
            rightHandOld = new Vector3(float.Parse(fields[39], culture) - xhipPos, float.Parse(fields[40], culture) - yhipPos, float.Parse(fields[41], culture) - zhipPos);
            rightArmOld = new Vector3(float.Parse(fields[36], culture) - xhipPos, float.Parse(fields[37], culture) - yhipPos, float.Parse(fields[38], culture) - zhipPos);
            hipOld = new Vector3(float.Parse(fields[0], culture) - xhipPos, float.Parse(fields[1], culture) - yhipPos, float.Parse(fields[2], culture) - zhipPos);

        }
    }
    void Update()
    {
        readCSV();
        frame+=1;
     
    }

    void LateUpdate()
    {
    }
    void readCSV()
    {
            var culture = (CultureInfo)CultureInfo.CurrentCulture.Clone();
            culture.NumberFormat.NumberDecimalSeparator = ".";

            string[] fields = records[frame].Split(';');

		// Set new body joint locations, draw new body parts between them
        if (fields.Length == 154)
        {
            string[] orig = records[1].Split(';');

            xhipPos = float.Parse(orig[0], culture);
            yhipPos = float.Parse(orig[1], culture);
            zhipPos = float.Parse(orig[2], culture);
            
             headNew = new Vector3(float.Parse(fields[35], culture) - xhipPos, float.Parse(fields[36], culture) - yhipPos, float.Parse(fields[37], culture) - zhipPos);
             neckNew = new Vector3(float.Parse(fields[28], culture) - xhipPos, float.Parse(fields[29], culture) - yhipPos, float.Parse(fields[30], culture) - zhipPos);
             abdomenNew = new Vector3(float.Parse(fields[7], culture) - xhipPos, float.Parse(fields[8], culture) - yhipPos, float.Parse(fields[9], culture) - zhipPos);
             chestNew = new Vector3(float.Parse(fields[14], culture) - xhipPos, float.Parse(fields[15], culture) - yhipPos, float.Parse(fields[16], culture) - zhipPos);
             torsoNew = new Vector3(float.Parse(fields[21], culture) - xhipPos, float.Parse(fields[22], culture) - yhipPos, float.Parse(fields[23], culture) - zhipPos);
             leftShoulderNew = new Vector3(float.Parse(fields[42], culture) - xhipPos, float.Parse(fields[43], culture) - yhipPos, float.Parse(fields[44], culture) - zhipPos);
             rightShoulderNew = new Vector3(float.Parse(fields[70], culture) - xhipPos, float.Parse(fields[71], culture) - yhipPos, float.Parse(fields[72], culture) - zhipPos);
             leftElbowNew = new Vector3(float.Parse(fields[49], culture) - xhipPos, float.Parse(fields[50], culture) - yhipPos, float.Parse(fields[51], culture) - zhipPos);
             leftArmNew = new Vector3(float.Parse(fields[56], culture) - xhipPos, float.Parse(fields[57], culture) - yhipPos, float.Parse(fields[58], culture) - zhipPos);
             rightElbowNew = new Vector3(float.Parse(fields[77], culture) - xhipPos, float.Parse(fields[78], culture) - yhipPos, float.Parse(fields[79], culture) - zhipPos);
             leftHandNew = new Vector3(float.Parse(fields[63], culture) - xhipPos, float.Parse(fields[64], culture) - yhipPos, float.Parse(fields[65], culture) - zhipPos);
             rightHandNew = new Vector3(float.Parse(fields[91], culture) - xhipPos, float.Parse(fields[92], culture) - yhipPos, float.Parse(fields[93], culture) - zhipPos);
             rightArmNew = new Vector3(float.Parse(fields[84], culture) - xhipPos, float.Parse(fields[85], culture) - yhipPos, float.Parse(fields[86], culture) - zhipPos);
             hipNew = new Vector3(float.Parse(fields[0], culture) - xhipPos, float.Parse(fields[1], culture) - yhipPos, float.Parse(fields[2], culture) - zhipPos);

          
        }
        else
        {
            string[] orig = records[0].Split(';');

            xhipPos = float.Parse(fields[0], culture);
            yhipPos = float.Parse(fields[1], culture);
            zhipPos = float.Parse(fields[2], culture);
            headNew = new Vector3(float.Parse(fields[15], culture) - xhipPos, float.Parse(fields[16], culture) - yhipPos, float.Parse(fields[17], culture) - zhipPos);
             neckNew = new Vector3(float.Parse(fields[12], culture) - xhipPos, float.Parse(fields[13], culture) - yhipPos, float.Parse(fields[14], culture) - zhipPos);
             abdomenNew = new Vector3(float.Parse(fields[3], culture) - xhipPos, float.Parse(fields[4], culture) - yhipPos, float.Parse(fields[5], culture) - zhipPos);
             chestNew = new Vector3(float.Parse(fields[6], culture) - xhipPos, float.Parse(fields[7], culture) - yhipPos, float.Parse(fields[8], culture) - zhipPos);
             torsoNew = new Vector3(float.Parse(fields[9], culture) - xhipPos, float.Parse(fields[10], culture) - yhipPos, float.Parse(fields[11], culture) - zhipPos);
             leftShoulderNew = new Vector3(float.Parse(fields[18], culture) - xhipPos, float.Parse(fields[19], culture) - yhipPos, float.Parse(fields[20], culture) - zhipPos);
             rightShoulderNew = new Vector3(float.Parse(fields[30], culture) - xhipPos, float.Parse(fields[31], culture) - yhipPos, float.Parse(fields[32], culture) - zhipPos);
             leftElbowNew = new Vector3(float.Parse(fields[21], culture) - xhipPos, float.Parse(fields[22], culture) - yhipPos, float.Parse(fields[23], culture) - zhipPos);
             leftArmNew = new Vector3(float.Parse(fields[24], culture) - xhipPos, float.Parse(fields[25], culture) - yhipPos, float.Parse(fields[26], culture) - zhipPos);
             rightElbowNew = new Vector3(float.Parse(fields[33], culture) - xhipPos, float.Parse(fields[34], culture) - yhipPos, float.Parse(fields[35], culture) - zhipPos);
             leftHandNew = new Vector3(float.Parse(fields[27], culture) - xhipPos, float.Parse(fields[28], culture) - yhipPos, float.Parse(fields[29], culture) - zhipPos);
             rightHandNew = new Vector3(float.Parse(fields[39], culture) - xhipPos, float.Parse(fields[40], culture) - yhipPos, float.Parse(fields[41], culture) - zhipPos);
             rightArmNew = new Vector3(float.Parse(fields[36], culture) - xhipPos, float.Parse(fields[37], culture) - yhipPos, float.Parse(fields[38], culture) - zhipPos);
             hipNew = new Vector3(float.Parse(fields[0], culture) - xhipPos, float.Parse(fields[1], culture) - yhipPos, float.Parse(fields[2], culture) - zhipPos);

        }

       


        head = Vector3.MoveTowards(headOld, headNew, 0.01f);
        neck = Vector3.MoveTowards(neckOld, neckNew, 0.01f);
        abdomen = Vector3.MoveTowards(abdomenOld, abdomenNew, 0.01f);
        chest = Vector3.MoveTowards(chestOld, chestNew, 0.01f);
        torso = Vector3.MoveTowards(torsoOld, torsoNew, 0.01f);
        leftShoulder = Vector3.MoveTowards(leftShoulderOld, leftShoulderNew, 0.01f);
        rightShoulder = Vector3.MoveTowards(rightShoulderOld, rightShoulderNew, 0.01f);
        leftElbow = Vector3.MoveTowards(leftElbowOld, leftElbowNew, 0.01f);
        rightElbow = Vector3.MoveTowards(rightElbowOld, rightElbowNew, 0.01f);
        leftArm = Vector3.MoveTowards(leftArmOld, leftArmNew, 0.01f);
        rightArm = Vector3.MoveTowards(rightArmOld, rightArmNew, 0.01f);
        leftHand = Vector3.MoveTowards(leftHandOld, leftHandNew, 0.01f);
        rightHand = Vector3.MoveTowards(rightHandOld, rightHandNew, 0.01f);
        hip = Vector3.MoveTowards(hipOld, hipNew, 0.01f);

         
       
		// Interpolate between new and old position for smooth transition

        headNeck.transform.position = Vector3.Lerp(head, neck, (float)0.5);
        headNeck.transform.LookAt(neck);
        headNeck.transform.localScale = new Vector3(60f, 70f, Vector3.Distance(head, neck));
    
        neckTorso.transform.position = Vector3.Lerp(neck, torso, (float)0.5);
        neckTorso.transform.LookAt(torso);
        neckTorso.transform.localScale = new Vector3(20f, 30f, Vector3.Distance(neck, torso));

        torsoChest.transform.position = Vector3.Lerp(torso, chest, (float)0.5);
        torsoChest.transform.LookAt(chest);
        torsoChest.transform.localScale = new Vector3(60f, 100f, Vector3.Distance(torso, chest));

        chestAbdomen.transform.position = Vector3.Lerp(chest, abdomen, (float)0.5);
        chestAbdomen.transform.LookAt(abdomen);
        chestAbdomen.transform.localScale = new Vector3(60f, 90f, Vector3.Distance(chest, abdomen));
        

        abdomenHip.transform.position = Vector3.Lerp(abdomen, hip, (float)0.5);
        abdomenHip.transform.LookAt(hip);
        abdomenHip.transform.localScale = new Vector3(40f, 105f, Vector3.Distance(abdomen, hip));

        torsoLShoulder.transform.position = Vector3.Lerp(torso, leftShoulder, (float)0.5);
        torsoLShoulder.transform.LookAt(leftShoulder);
        torsoLShoulder.transform.localScale = new Vector3(10f, 10f, Vector3.Distance(torso, leftShoulder));

        torsoRShoulder.transform.position = Vector3.Lerp(torso, rightShoulder, .5f);
        torsoRShoulder.transform.LookAt(rightShoulder);
        torsoRShoulder.transform.localScale = new Vector3(10f, 10f, Vector3.Distance(torso, rightShoulder));

        LShoulderElbow.transform.position = Vector3.Lerp(leftShoulder, leftElbow, .5f);
        LShoulderElbow.transform.LookAt(leftElbow);
        LShoulderElbow.transform.localScale = new Vector3(20f, 20f, Vector3.Distance(leftShoulder, leftElbow));

        LElbowArm.transform.position = Vector3.Lerp(leftElbow, leftArm, .5f);
        LElbowArm.transform.LookAt(leftArm);
        LElbowArm.transform.localScale = new Vector3(20f, 20f, Vector3.Distance(leftElbow, leftArm));

        LArmHand.transform.position = Vector3.Lerp(leftArm, leftHand, .5f);
        LArmHand.transform.LookAt(leftHand);
        LArmHand.transform.localScale = new Vector3(20f, 35f, Vector3.Distance(leftArm, leftHand));

        RShoulderElbow.transform.position = Vector3.Lerp(rightShoulder, rightElbow, .5f);
        RShoulderElbow.transform.LookAt(rightElbow);
        RShoulderElbow.transform.localScale = new Vector3(20f, 20f, Vector3.Distance(rightShoulder, rightElbow));

        RElbowArm.transform.position = Vector3.Lerp(rightElbow, rightArm, .5f);
        RElbowArm.transform.LookAt(rightArm);
        RElbowArm.transform.localScale = new Vector3(20f, 20f, Vector3.Distance(rightElbow, rightArm));

        RArmHand.transform.position = Vector3.Lerp(rightArm, rightHand, .5f);
        RArmHand.transform.LookAt(rightHand);
        RArmHand.transform.localScale = new Vector3(20f, 35f, Vector3.Distance(rightArm, rightHand));

 

        RShoulder.transform.position = rightShoulder;
        RShoulder.transform.localScale = new Vector3(50f, 50f, 50f);

        LShoulder.transform.position = leftShoulder;
        LShoulder.transform.localScale = new Vector3(50f, 50f, 50f);

        RElbow.transform.position = rightElbow;
        RElbow.transform.localScale = new Vector3(50f, 50f, 50f);

        LElbow.transform.position = leftElbow;
        LElbow.transform.localScale = new Vector3(50f, 50f, 50f);
        

        RHand.transform.position = rightArm;
        RHand.transform.localScale = new Vector3(40f, 40f, 40f);

        LHand.transform.position = leftArm;
        LHand.transform.localScale = new Vector3(40f, 40f, 40f);


        Chest.transform.position = chest;
        Chest.transform.localScale = new Vector3(25f, 160f, 50f);

        Torso.transform.position = torso;
        Torso.transform.localScale = new Vector3(18f, 70f, 30f);

        Abdomen.transform.position = abdomen;
        Abdomen.transform.localScale = new Vector3(18f, 70f, 30f);

        Hip.transform.position = hip;
        Hip.transform.localScale = new Vector3(18f, 70f, 30f);



        headOld = headNew;
            neckOld = neckNew;
            abdomenOld = abdomenNew;
            torsoOld = torsoNew;
            chestOld = chestNew;
            leftShoulderOld = leftShoulderNew;
            rightShoulderOld = rightShoulderNew;
            leftElbowOld = leftElbowNew;
            rightElbowOld = rightElbowNew;
            leftArmOld = leftArmNew;
            rightArmOld = rightArmNew;
            leftHandOld = leftHandNew;
            rightHandOld = rightHandNew;
        hipOld = hipNew;

        }
  
    }
