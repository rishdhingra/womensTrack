#!/bin/bash
# EndoDetect AI - One-Click Starter Script
# This automates the entire setup process

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║          EndoDetect AI - Automated Setup                 ║"
echo "║          RWJ Women's Health Pitch - Jan 23, 2026         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$HOME/EndoDetect-AI"
cd "$PROJECT_DIR"

# Step 1: Check Python
echo -e "${BLUE}[1/8] Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✅ Found: $PYTHON_VERSION${NC}"
echo ""

# Step 2: Create virtual environment
echo -e "${BLUE}[2/8] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi
source venv/bin/activate
echo ""

# Step 3: Install dependencies
echo -e "${BLUE}[3/8] Installing Python dependencies...${NC}"
echo "This may take 5-10 minutes..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Step 4: Check AWS CLI
echo -e "${BLUE}[4/8] Checking AWS CLI...${NC}"
if ! command -v aws &> /dev/null; then
    echo -e "${YELLOW}⚠️  AWS CLI not found. Installing...${NC}"
    if command -v brew &> /dev/null; then
        brew install awscli
        echo -e "${GREEN}✅ AWS CLI installed${NC}"
    else
        echo -e "${RED}❌ Homebrew not found. Please install AWS CLI manually${NC}"
        echo "Visit: https://aws.amazon.com/cli/"
    fi
else
    echo -e "${GREEN}✅ AWS CLI already installed${NC}"
fi
echo ""

# Step 5: Configure AWS (interactive)
echo -e "${BLUE}[5/8] Configuring AWS credentials...${NC}"
if aws sts get-caller-identity &> /dev/null; then
    echo -e "${GREEN}✅ AWS credentials already configured${NC}"
else
    echo -e "${YELLOW}Please enter your AWS credentials:${NC}"
    aws configure
    if aws sts get-caller-identity &> /dev/null; then
        echo -e "${GREEN}✅ AWS credentials configured successfully${NC}"
    else
        echo -e "${RED}❌ AWS configuration failed. Continuing without AWS...${NC}"
    fi
fi
echo ""

# Step 6: Create directories
echo -e "${BLUE}[6/8] Creating project directories...${NC}"
mkdir -p data models demo_outputs
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Step 7: Download dataset
echo -e "${BLUE}[7/8] Downloading UT-EndoMRI dataset...${NC}"
if [ ! -f "data/dataset.zip" ]; then
    echo "Downloading from Zenodo (this may take 10-15 minutes)..."
    cd data
    curl -L "https://zenodo.org/records/15750762/files/UT-EndoMRI.zip" -o dataset.zip
    
    if [ -f "dataset.zip" ]; then
        echo "Extracting dataset..."
        unzip -q dataset.zip
        echo -e "${GREEN}✅ Dataset downloaded and extracted${NC}"
    else
        echo -e "${RED}❌ Download failed. Please download manually${NC}"
    fi
    cd ..
else
    echo -e "${YELLOW}⚠️  Dataset already exists${NC}"
fi
echo ""

# Step 8: Start training decision
echo -e "${BLUE}[8/8] Ready to start training!${NC}"
echo ""
echo "You have two options:"
echo ""
echo -e "${GREEN}Option A: Local CPU Training (Slower, ~24 hours)${NC}"
echo "  Command:"
echo "  python train_segmentation_model.py \\"
echo "    --data_dir ./data/D2_TCPW \\"
echo "    --output_dir ./models \\"
echo "    --epochs 50 \\"
echo "    --batch_size 2 \\"
echo "    --device cpu"
echo ""
echo -e "${GREEN}Option B: AWS EC2 GPU Training (Faster, ~4-6 hours)${NC}"
echo "  1. Run: ./setup_aws.sh"
echo "  2. Launch EC2 instance (see QUICK_START_GUIDE.md)"
echo "  3. SSH to instance and run training there"
echo ""
read -p "Start local CPU training now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Starting training... This will run overnight.${NC}"
    echo "You can safely close this terminal - training will continue."
    echo ""
    
    nohup python train_segmentation_model.py \
        --data_dir ./data/D2_TCPW \
        --output_dir ./models \
        --epochs 50 \
        --batch_size 2 \
        --device cpu > training.log 2>&1 &
    
    TRAIN_PID=$!
    echo -e "${GREEN}✅ Training started (PID: $TRAIN_PID)${NC}"
    echo "Monitor progress: tail -f training.log"
    echo ""
else
    echo -e "${YELLOW}Skipping training. You can start it later.${NC}"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo "1. Wait for training to complete (check: tail -f training.log)"
echo "2. Generate demo outputs: python generate_demo_outputs.py --model_path ./models/best_model.pth --data_dir ./data/D2_TCPW"
echo "3. Create pitch deck using generated visualizations"
echo "4. Practice presentation (must be ≤5 minutes!)"
echo ""
echo -e "${BLUE}Important Files:${NC}"
echo "  📖 README.md - Start here for overview"
echo "  📋 QUICK_START_GUIDE.md - Detailed instructions"
echo "  📊 demo_outputs/ - Generated visualizations (after step 2)"
echo ""
echo -e "${GREEN}Good luck with your pitch on Jan 23! 🚀${NC}"
