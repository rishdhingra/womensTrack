#!/bin/bash
# Quick training status check

echo "=========================================="
echo "Training Status Check"
echo "=========================================="
echo ""

# Check if training process is running
if pgrep -f "train_radiomics_simple" > /dev/null; then
    echo "✅ Training process: RUNNING"
    ps aux | grep train_radiomics_simple | grep -v grep | awk '{print "   PID: " $2 ", CPU: " $3 "%, Memory: " $4 "%"}'
else
    echo "❌ Training process: NOT RUNNING"
fi
echo ""

# Check output directory
if [ -d "./models_radiomics" ]; then
    echo "📁 Output directory: EXISTS"
    echo "   Files:"
    ls -lh ./models_radiomics/ 2>/dev/null | tail -n +2 | awk '{print "   " $9 " (" $5 ")"}'
    
    # Check for history file
    if [ -f "./models_radiomics/history.json" ]; then
        echo ""
        echo "📈 Training Progress:"
        python3 << EOF
import json
from pathlib import Path

try:
    with open('./models_radiomics/history.json', 'r') as f:
        history = json.load(f)
    epochs = len(history.get('train_acc', []))
    if epochs > 0:
        print(f"   Epochs completed: {epochs}")
        print(f"   Latest Train Acc: {history['train_acc'][-1]:.4f}")
        if len(history.get('val_acc', [])) > 0:
            print(f"   Latest Val Acc: {history['val_acc'][-1]:.4f}")
    else:
        print("   Feature extraction phase...")
except:
    print("   Feature extraction phase...")
EOF
    else
        echo ""
        echo "⏳ Status: Feature extraction phase (no training history yet)"
    fi
    
    # Check for metadata
    if [ -f "./models_radiomics/metadata.json" ]; then
        echo ""
        echo "📋 Model Info:"
        python3 << EOF
import json
try:
    with open('./models_radiomics/metadata.json', 'r') as f:
        meta = json.load(f)
    print(f"   Dataset: {meta.get('dataset', 'N/A')}")
    print(f"   Features: {meta.get('radiomics_features', 'N/A')}")
    if 'best_accuracy_percentage' in meta:
        print(f"   Best Accuracy: {meta['best_accuracy_percentage']:.2f}%")
except:
    pass
EOF
    fi
else
    echo "📁 Output directory: NOT CREATED YET"
    echo "   Status: Feature extraction phase..."
fi

echo ""
echo "=========================================="
