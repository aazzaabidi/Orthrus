# main.py
import argparse
from train import train_model
from models import resnet50, cnn1D

def main(args):
    if args.model == 'single-technique':
        model = resnet50(input_shape=input_shape, nb_classes=nb_classes)
    elif args.model == 'combined-techniques':
        model = cnn1D(input_shape=input_shape2, nb_classes=nb_classes)
    else:
        raise ValueError("Invalid model name. Please choose either 'single-technique' or 'combined-techniques'.")

    # Train model
    train_model(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train land use and land cover mapping models.")
    parser.add_argument('--model', type=str, default='single-technique', help="Choose model: 'single-technique' or 'combined-techniques'")
    args = parser.parse_args()
    
    main(args)
