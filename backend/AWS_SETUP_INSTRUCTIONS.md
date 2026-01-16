# AWS Setup - REQUIRED for Real Demo

## 🚨 WE NEED AWS FOR THE PITCH!

You're right - we need AWS to:
1. Train models on GPU (4-6 hours vs 24+ CPU)
2. Store real medical imaging data
3. Generate actual segmentation outputs
4. Connect frontend to real backend

---

## Step 1: Get Your AWS Credentials (DO THIS NOW)

### Option A: If You Have AWS Account Already
1. Go to: https://console.aws.amazon.com
2. Sign in
3. Click your name (top right) → Security Credentials
4. Scroll to "Access Keys"
5. Click "Create Access Key"
6. Download the CSV file (has Access Key ID + Secret Key)

### Option B: If You DON'T Have AWS Account
1. Go to: https://aws.amazon.com/free
2. Click "Create a Free Account"
3. Follow signup (needs credit card but won't charge for our usage)
4. After signup, follow Option A steps above

### Option C: Use Rutgers AWS Credits (If Available)
Ask Dr. Yanamala or IT if Rutgers has AWS research credits

---

## Step 2: Configure AWS CLI (Run This After Getting Credentials)

```bash
cd ~/EndoDetect-AI
aws configure
```

**When prompted, enter:**
- AWS Access Key ID: [paste from your CSV]
- AWS Secret Access Key: [paste from your CSV]
- Default region: us-east-1
- Default output format: json

---

## Step 3: Run Automated Setup

```bash
cd ~/EndoDetect-AI
./setup_aws.sh
```

This will:
- Create encrypted S3 bucket
- Set up IAM roles
- Configure security groups
- Upload sample data

---

## Step 4: Launch GPU Training

```bash
# Option A: SageMaker (Easier)
cd ~/EndoDetect-AI
python launch_sagemaker_training.py

# Option B: EC2 (More control)
# Use AWS Console to launch g4dn.xlarge
# Then SSH and run training
```

---

## Cost Estimate

### For Jan 23 Demo:
- S3 storage: $0.23
- EC2 GPU (6 hours): $3.16
- Lambda/API: $0.50
**Total: ~$4**

### AWS Free Tier Includes:
- 5GB S3 storage (free)
- Lambda calls (1M free/month)
- Some EC2 compute (limited)

---

## Security Notes

✅ All data is de-identified (no PHI)
✅ S3 encrypted (AES-256)
✅ No public access
✅ IAM roles for access control
✅ MFA recommended

---

## What We'll Create

1. **S3 Bucket:** s3://endodetect-ai-rwjms
2. **EC2 Instance:** g4dn.xlarge (NVIDIA T4 GPU)
3. **Lambda Functions:**
   - upload_image
   - trigger_training
   - get_results
4. **API Gateway:** REST API for frontend
5. **DynamoDB:** Experiment tracking

---

## Timeline

**Tonight (Jan 14):**
- [ ] Get AWS credentials
- [ ] Configure AWS CLI
- [ ] Run setup_aws.sh

**Tomorrow (Jan 15):**
- [ ] Launch GPU training (4-6 hours)
- [ ] Upload datasets to S3
- [ ] Test inference pipeline

**Jan 16-17:**
- [ ] Generate demo outputs
- [ ] Integrate frontend with API
- [ ] Test end-to-end flow

---

## Need Help?

**Getting credentials:** Check your email for AWS signup confirmation
**Configuration issues:** Run `aws sts get-caller-identity` to test
**Cost concerns:** We can set billing alerts at $10

---

## NEXT STEP: Get your AWS credentials and run this:

```bash
aws configure
```

Then tell me when done and I'll provision everything! 🚀
