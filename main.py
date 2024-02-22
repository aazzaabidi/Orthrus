import argparse
from models import inceptiontime, multirocket, RandomForest, resnet50

# Define a function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train different models")
    parser.add_argument("model", type=str, choices=["inceptiontime", "multirocket", "RandomForest", "resnet50"],
                        help="Choose which model to use: inceptiontime, multirocket, RandomForest, or resnet50")
    return parser.parse_args()

# Load the data and perform any necessary preprocessing

# Parse command line arguments
args = parse_args()

# Train the selected model
if args.model == "inceptiontime":
    # Train InceptionTime model
    inceptiontime.train_model()
elif args.model == "multirocket":
    # Train MultiRocket model
    multirocket.train_model()
elif args.model == "RandomForest":
    # Train RandomForest model
    RandomForest.train_model()
elif args.model == "resnet50":
    # Train ResNet50 model
    resnet50.train_model()
