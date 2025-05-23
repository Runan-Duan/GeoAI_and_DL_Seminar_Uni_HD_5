import argparse
import torch
from trainer import RoadSurfaceTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='convnext_small', type=str,
                   choices=['resnet50', 'efficientnet_b4', 'convnext_small'],
                   help='Model architecture to use')
    parser.add_argument('--models_dir', default='./models', type=str, help='path to model')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=120, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--data_path', default='./data/raw/StreetSurfaceVis/s_1024', type=str, help='dataset path')
    parser.add_argument('--logs_dir', default='./logs', type=str, help='logs path')
    args = parser.parse_args()

    print(f"Using Cuda: {torch.cuda.is_available()}")
    trainer = RoadSurfaceTrainer(args)
    trainer.train()
    trainer.test(confusion_matrix=True)


if __name__ == "__main__":
    main()
