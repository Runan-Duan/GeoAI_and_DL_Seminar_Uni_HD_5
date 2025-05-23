import argparse
import torch
from visualizer import ModelVisualizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='convnext_small', type=str,
                   choices=['resnet50', 'efficientnet_b4', 'convnext_small'],
                   help='Model architecture to use')
    parser.add_argument('--models_dir', default='./models', type=str, help='path to model')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--data_path', default='./data/raw/StreetSurfaceVis/s_1024', type=str, help='dataset path')
    args = parser.parse_args()

    print(f"Using Cuda: {torch.cuda.is_available()}")
    visualizer = ModelVisualizer(args)
    
    # Run visualizations
    visualizer.test(confusion_matrix=True, visualization=False, run_gradcam=True)

if __name__ == "__main__":
    main()
