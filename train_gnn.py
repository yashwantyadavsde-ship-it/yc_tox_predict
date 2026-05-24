import os
import sys

# Ensure model directory is on path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model.train import train_and_save

def main():
    print("Welcome to the Toxicity GNN Training Pipeline!")
    print("Executing GDL calibration on reference compound sets...")
    train_and_save()
    print("\nTraining Complete! GNN backbone weights successfully saved under model/weights/")

if __name__ == '__main__':
    main()
