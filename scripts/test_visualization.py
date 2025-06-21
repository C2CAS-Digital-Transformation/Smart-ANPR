#!/usr/bin/env python3
"""
Test Training Visualization
===========================

This script demonstrates the training visualization without running actual training.
Useful for testing the visualization components.
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src" / "training"))

# Import the visualizer (this will work if we run from project root)
try:
    from src.training.train_ocr import TrainingVisualizer
    print("✓ Successfully imported TrainingVisualizer")
except ImportError as e:
    print(f"✗ Failed to import TrainingVisualizer: {e}")
    print("Please run this script from the ANPR project root directory")
    sys.exit(1)

def simulate_training():
    """Simulate a training session with realistic data"""
    print("🎨 Initializing Training Visualization Demo")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = TrainingVisualizer(target_accuracy=0.95)
    
    print("📊 Simulating training progress...")
    print("This will show how the visualization looks during actual training")
    print("Close the plot window to stop the simulation")
    
    try:
        # Simulate 20 epochs of training
        for epoch in range(1, 21):
            print(f"Simulating Epoch {epoch}/20...")
            
            # Simulate realistic training metrics
            # Start with poor performance and gradually improve
            base_acc = 0.3 + (epoch / 20) * 0.6  # Gradually improve from 30% to 90%
            noise = np.random.normal(0, 0.02)  # Add some noise
            val_accuracy = min(0.98, max(0.1, base_acc + noise))
            
            # Character accuracy is usually higher
            char_accuracy = min(0.99, val_accuracy + 0.05 + np.random.normal(0, 0.01))
            
            # Training loss decreases over time
            train_loss = 3.0 * np.exp(-epoch * 0.15) + np.random.normal(0, 0.1)
            train_loss = max(0.1, train_loss)
            
            # Validation loss
            val_loss = train_loss + np.random.normal(0, 0.05)
            val_loss = max(0.1, val_loss)
            
            # Learning rate with OneCycleLR pattern
            if epoch <= 10:
                lr = 1e-4 + (1e-3 - 1e-4) * (epoch / 10)  # Increase
            else:
                lr = 1e-3 * np.exp(-(epoch - 10) * 0.2)  # Decrease
            
            # Update epoch data
            visualizer.update_epoch_data(epoch, train_loss, val_loss, val_accuracy, char_accuracy, lr)
            
            # Simulate batch losses for this epoch
            num_batches = 50
            for batch in range(num_batches):
                global_batch = (epoch - 1) * num_batches + batch
                # Batch loss with some variation
                batch_loss = train_loss + np.random.normal(0, 0.3)
                batch_loss = max(0.1, batch_loss)
                visualizer.update_batch_loss(global_batch, batch_loss)
                
                # Update plots every 10 batches
                if batch % 10 == 0:
                    visualizer.update_plots()
                    time.sleep(0.1)  # Small delay to see animation
            
            # Update plots at end of epoch
            visualizer.update_plots()
            
            # Print progress
            print(f"  Epoch {epoch}: Val Acc = {val_accuracy*100:.1f}%, "
                  f"Char Acc = {char_accuracy*100:.1f}%, "
                  f"Train Loss = {train_loss:.3f}")
            
            # Check if target achieved
            if val_accuracy >= 0.95:
                print(f"🎯 TARGET ACHIEVED at epoch {epoch}!")
            
            # Pause between epochs
            time.sleep(0.5)
        
        print("\n🎉 Simulation Complete!")
        print("The visualization shows what you'll see during actual training.")
        print("Keep the plot window open to examine the results.")
        print("Press Enter to close and exit...")
        input()
        
    except KeyboardInterrupt:
        print("\n⚠️ Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
    finally:
        # Clean up
        visualizer.close()
        plt.close('all')
        print("✓ Visualization closed")

def main():
    """Main function"""
    print("ANPR OCR Training Visualization Demo")
    print("=" * 40)
    print()
    print("This demo shows what the training visualization looks like.")
    print("It simulates realistic training progress over 20 epochs.")
    print()
    
    # Check if matplotlib backend works
    try:
        plt.figure()
        plt.close()
        print("✓ Matplotlib backend working")
    except Exception as e:
        print(f"❌ Matplotlib backend issue: {e}")
        print("You may need to install tkinter: pip install tk")
        return
    
    response = input("Start visualization demo? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        simulate_training()
    else:
        print("Demo cancelled.")

if __name__ == "__main__":
    main() 