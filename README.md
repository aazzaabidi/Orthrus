[orthus.pdf](https://github.com/aazzaabidi/Orthrus/files/14377893/orthus.pdf)# Orthrus
This code representes **Orthrus**,  a novel framework for land cover mapping that leverages multi-scale information and 2D encoding techniques to cope with satellite image time series information. The approach is depicted in  Figure~\ref{fig:method}, which sketches the flowchart of the process. The proposed framework consists of two stages: the first stage performs 2D Encoding using various techniques for each of the multi-scale information, while the second stage uses a dual branch Convolutional Neural Network to classify the encoded time series data. This framework  effectively captures both the pixel-level and object-level features of multivariate time series data, improving the accuracy of classification tasks.

![Screenshot-105](https://github.com/aazzaabidi/Orthrus/assets/73762433/adae2278-a540-4bb5-8b35-683089c04c0a)



### Usage

To use this code, follow the instructions below:

1. **Dataset Preparation**:
   - Ensure that the datasets are split and encoded using the provided script `2D_encode.py`.
   - Place the split and encoded datasets in the specified paths for training, validation, and testing.

2. **Model Configuration**:
   - Modify the `input_pxl` and `input_obj` variables according to the input shape for pixel and object classification branches.
   - Configure the number of classes (`nb_classes`) based on your dataset.

3. **Model Training**:
   - Configure the hyperparameters such as batch size (`BATCH_SIZE`) and number of epochs (`EPOCHS`).
   - Execute the code to train the model using the prepared datasets.

4. **Model Evaluation**:
   - After training, the model is evaluated on the validation and test sets.
   - Evaluation metrics such as accuracy, F1-score, and Cohen's Kappa are calculated.
   - Confusion matrix and classification reports are generated for further analysis.

### Requirements

Ensure you have the following dependencies installed:

- TensorFlow
- Keras
- NumPy
- tqdm
- scikit-learn
- pandas
- seaborn

### File Structure

- `2D_encode.py`: Script for dataset splitting and encoding.
- `README.md`: This file providing information and instructions.
- `residual_2D_model.py`: Main code file containing the implementation of the ResNet-50 model.
- `requirements.txt`: File listing all required dependencies for easy installation.

### Acknowledgments

This code is adapted from various sources, including TensorFlow/Keras documentation and ResNet papers. Special thanks to the TensorFlow and Keras communities for providing valuable resources and support.

For any questions or issues, please contact [author's name/email].
