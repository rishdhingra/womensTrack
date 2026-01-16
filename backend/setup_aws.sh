#!/bin/bash
# EndoDetect AI - AWS Setup Script
# Run this after installing AWS CLI and configuring credentials

set -e  # Exit on error

echo "=== EndoDetect AI - AWS Setup ==="
echo ""

# Configuration
BUCKET_NAME="endodetect-ai-rwjms"
REGION="us-east-1"
DATA_DIR="$HOME/EndoDetect-AI/data"

# Check AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Installing..."
    brew install awscli
fi

# Check AWS credentials
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured"
    echo "Please run: aws configure"
    exit 1
fi

echo "✅ AWS credentials valid"
echo ""

# Create S3 bucket
echo "Creating S3 bucket: $BUCKET_NAME"
if aws s3 mb s3://$BUCKET_NAME --region $REGION 2>/dev/null; then
    echo "✅ Bucket created"
else
    echo "⚠️  Bucket already exists or error occurred"
fi

# Enable encryption
echo "Enabling server-side encryption..."
aws s3api put-bucket-encryption \
  --bucket $BUCKET_NAME \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }' 2>/dev/null || echo "⚠️  Encryption already enabled"

echo "✅ Bucket encryption enabled"
echo ""

# Block public access
echo "Blocking public access..."
aws s3api put-public-access-block \
  --bucket $BUCKET_NAME \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" \
    2>/dev/null || echo "⚠️  Public access already blocked"

echo "✅ Public access blocked"
echo ""

# Upload data if it exists
if [ -d "$DATA_DIR" ]; then
    echo "Uploading data to S3..."
    aws s3 sync $DATA_DIR s3://$BUCKET_NAME/datasets/ \
      --exclude "*.zip" \
      --exclude "*.tar.gz"
    echo "✅ Data uploaded"
else
    echo "⚠️  Data directory not found: $DATA_DIR"
    echo "Download dataset first using the quick start guide"
fi

echo ""
echo "=== AWS Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Download dataset: curl -L 'https://zenodo.org/records/15750762/files/UT-EndoMRI.zip' -o data/dataset.zip"
echo "2. Launch EC2 GPU instance for training"
echo "3. Run training script"
echo ""
echo "S3 Bucket: s3://$BUCKET_NAME"
echo "Region: $REGION"
