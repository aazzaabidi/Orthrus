# Orthrus
This code representes **Orthrus**,  a novel framework for land cover mapping that leverages multi-scale information and 2D encoding techniques to cope with satellite image time series information. The approach is depicted in  Figure~\ref{fig:method}, which sketches the flowchart of the process. The proposed framework consists of two stages: the first stage performs 2D Encoding using various techniques for each of the multi-scale information, while the second stage uses a dual branch Convolutional Neural Network to classify the encoded time series data. This framework  effectively captures both the pixel-level and object-level features of multivariate time series data, improving the accuracy of classification tasks.

![Screenshot-105](https://github.com/aazzaabidi/Orthrus/assets/73762433/adae2278-a540-4bb5-8b35-683089c04c0a)

```

## Overview
This repository contains scripts to train different models for a given task. The available models include InceptionTime, MultiRocket, RandomForest, and ResNet50.

## Requirements
- Python 3.x
- TensorFlow (for deep learning models)
- scikit-learn (for RandomForest)
- Other dependencies as required by individual models

## Usage
1. Clone this repository to your local machine.
2. Install the required dependencies using pip:
   ```
   pip install -r requirements.txt
   ```
3. Run the `main.py` script with the desired model as an argument to start training. Choose one of the following models:
   - inceptiontime
   - multirocket
   - RandomForest
   - resnet50
   For example:
   ```
   python main.py inceptiontime
   ```
4. The script will train the selected model using the provided data and display the training progress.
5. After training completes, the trained model will be saved or any other specified location depending on the model.

## File Structure
- `main.py`: Python script to initiate model training based on user input.
- `models.py`: Python module containing the implementations of different models.
- `data/`: Directory containing the dataset or links to the dataset used for training.
- `README.md`: This file.

## Additional Notes
- Make sure to adjust the data loading and preprocessing steps in the `main.py` script as per your dataset requirements.
- Each model may have its own specific parameters or configurations. Refer to the respective model implementation in `models.py` for more details.

```

You can include this README file in your project repository to provide instructions on how to use the `main.py` script for training different models. Adjust the content as needed based on your specific project requirements.
### Citation



For any questions or issues, please contact aazzaabidi@gmail.com.
