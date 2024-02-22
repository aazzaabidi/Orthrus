import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
import model from models

# Load the data
# Assuming the data is in a format similar to the following:
# X_train: 3D numpy array of shape (num_train_samples, img_height, img_width, num_channels)
# y_train: 1D numpy array of shape (num_train_samples,)
# X_test: 3D numpy array of shape (num_test_samples, img_height, img_width, num_channels)
# y_test: 1D numpy array of shape (num_test_samples,)




model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Create a checkpoint callback
checkpoint_callback = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

# Train the model
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=100,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint_callback])

# Evaluate the model
_, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

y_pred = np.argmax(model.predict(X_test), axis=-1)

# Calculate the metrics
f1 = f1_score(y_test, y_pred, average='weighted')
cohen_kappa = cohen_kappa_score(y_test, y_pred)

print('F1 score:', f1)
print('Cohen\'s kappa:', cohen_kappa)

# Create a confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
