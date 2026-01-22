#!/usr/bin/env python3
"""
Training Monitor - Real-time monitoring of radiomics training progress
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import subprocess

def get_process_info():
    """Get information about running training process"""
    try:
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, 
            text=True
        )
        lines = result.stdout.split('\n')
        for line in lines:
            if 'train_radiomics_simple' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    return {
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'status': parts[7],
                        'time': parts[9]
                    }
    except:
        pass
    return None

def check_training_output(output_dir):
    """Check for training output files"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    files = {
        'model': output_path / 'best_model.pth',
        'history': output_path / 'history.json',
        'metadata': output_path / 'metadata.json',
        'plot': output_path / 'training_history.png'
    }
    
    status = {}
    for name, path in files.items():
        status[name] = path.exists()
        if path.exists():
            status[f'{name}_size'] = path.stat().st_size
            status[f'{name}_modified'] = datetime.fromtimestamp(path.stat().st_mtime).strftime('%H:%M:%S')
    
    return status

def get_training_progress(output_dir):
    """Get training progress from history file"""
    history_path = Path(output_dir) / 'history.json'
    if not history_path.exists():
        return None
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = len(history.get('train_acc', []))
        if epochs > 0:
            latest_train_acc = history['train_acc'][-1]
            latest_val_acc = history['val_acc'][-1] if len(history.get('val_acc', [])) > 0 else 0
            latest_train_loss = history['train_loss'][-1]
            latest_val_loss = history['val_loss'][-1] if len(history.get('val_loss', [])) > 0 else 0
            
            return {
                'epochs_completed': epochs,
                'latest_train_acc': latest_train_acc,
                'latest_val_acc': latest_val_acc,
                'latest_train_loss': latest_train_loss,
                'latest_val_loss': latest_val_loss
            }
    except Exception as e:
        return {'error': str(e)}
    
    return None

def get_metadata(output_dir):
    """Get training metadata"""
    metadata_path = Path(output_dir) / 'metadata.json'
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except:
        return None

def monitor_training(output_dir='./models_radiomics', refresh_interval=5):
    """Monitor training progress"""
    print("=" * 80)
    print(" ENDODETECT AI - TRAINING MONITOR")
    print("=" * 80)
    print(f"Monitoring directory: {output_dir}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 80)
    print()
    
    start_time = time.time()
    last_epoch = 0
    
    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("=" * 80)
            print(" ENDODETECT AI - TRAINING MONITOR")
            print("=" * 80)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Elapsed: {int(time.time() - start_time)} seconds")
            print("=" * 80)
            print()
            
            # Check process
            process_info = get_process_info()
            if process_info:
                print("📊 PROCESS STATUS")
                print(f"  PID: {process_info['pid']}")
                print(f"  CPU: {process_info['cpu']}%")
                print(f"  Memory: {process_info['mem']}%")
                print(f"  Status: {process_info['status']}")
                print(f"  Runtime: {process_info['time']}")
                print()
            else:
                print("⚠️  PROCESS STATUS: Not found (may have completed)")
                print()
            
            # Check output files
            output_status = check_training_output(output_dir)
            if output_status:
                print("📁 OUTPUT FILES")
                for name in ['model', 'history', 'metadata', 'plot']:
                    if output_status.get(name):
                        size_kb = output_status.get(f'{name}_size', 0) / 1024
                        modified = output_status.get(f'{name}_modified', 'N/A')
                        print(f"  ✅ {name}: {size_kb:.1f} KB (modified: {modified})")
                    else:
                        print(f"  ⏳ {name}: Not created yet")
                print()
            else:
                print("📁 OUTPUT FILES: Directory not created yet (feature extraction phase)")
                print()
            
            # Check training progress
            progress = get_training_progress(output_dir)
            if progress:
                if 'error' in progress:
                    print(f"⚠️  Error reading progress: {progress['error']}")
                else:
                    epochs = progress['epochs_completed']
                    if epochs > last_epoch:
                        print(f"🎉 New epoch completed! (Total: {epochs})")
                        last_epoch = epochs
                    
                    print("📈 TRAINING PROGRESS")
                    print(f"  Epochs completed: {epochs}")
                    print(f"  Latest Train Accuracy: {progress['latest_train_acc']:.4f} ({progress['latest_train_acc']*100:.2f}%)")
                    print(f"  Latest Val Accuracy: {progress['latest_val_acc']:.4f} ({progress['latest_val_acc']*100:.2f}%)")
                    print(f"  Latest Train Loss: {progress['latest_train_loss']:.4f}")
                    print(f"  Latest Val Loss: {progress['latest_val_loss']:.4f}")
                    print()
            else:
                print("📈 TRAINING PROGRESS: Not started yet (feature extraction phase)")
                print()
            
            # Check metadata
            metadata = get_metadata(output_dir)
            if metadata:
                print("📋 METADATA")
                print(f"  Dataset: {metadata.get('dataset', 'N/A')}")
                print(f"  Feature dimension: {metadata.get('radiomics_features', 'N/A')}")
                print(f"  Train size: {metadata.get('train_size', 'N/A')}")
                print(f"  Val size: {metadata.get('val_size', 'N/A')}")
                if 'best_accuracy_percentage' in metadata:
                    print(f"  Best Accuracy: {metadata['best_accuracy_percentage']:.2f}%")
                print()
            
            print("=" * 80)
            print(f"Refreshing in {refresh_interval} seconds... (Ctrl+C to stop)")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("=" * 80)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--output_dir', type=str, default='./models_radiomics',
                       help='Output directory to monitor')
    parser.add_argument('--interval', type=int, default=5,
                       help='Refresh interval in seconds')
    args = parser.parse_args()
    
    monitor_training(args.output_dir, args.interval)
