import tensorflow as tf
from tensorflow.keras.layers import Input
import tensorflow.keras as tfk
from tensorflow.keras.layers import  Activation, Conv2D, MaxPooling2D,GlobalAveragePooling2D, Flatten,  Add,BatchNormalization,  Dense
from tensorflow.python.ops.gen_batch_ops import Batch
tf.keras.backend.set_floatx('float32')
import numpy as np
from tqdm import tqdm
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, cohen_kappa_score
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import time
from pandas import DataFrame
import seaborn as sns

#load the train,test,valid set splits for the pixel branch

x_train =np.load('path')
x_valid=np.load ('path')
x_test =np.load ('path')

#load the train,test,valid set splits for the pixel object

o_train =np.load('path')
o_valid=np.load ('path')
o_test =np.load ('path')

train_y=np.load ('path')
valid_y= np.load ('path')
test_y= np.load ('path')


encoder = LabelEncoder()
encoder.fit(train_y)
train_y_enc = encoder.transform(train_y)
valid_y_enc = encoder.transform(valid_y)
test_y_enc = encoder.transform(test_y)
np.unique(train_y), np.unique(train_y_enc), np.unique(valid_y), np.unique(valid_y_enc)



nb_classes=np.length(train_y)



def residual_block(X_start, filters,  reduce=False, res_conv2d=False):
    """
    Residual building block used by ResNet-50
    """

    nb_filters_1, nb_filters_2, nb_filters_3 = filters
    strides_1 = [2,2] if reduce else [1,1]
        
    X = Conv2D(filters=nb_filters_1, kernel_size=[1,1], strides=strides_1, padding='same')(X_start)
    X = BatchNormalization()(X)      # default axis-1 is ok
    X = Activation('relu')(X)
    
    X = Conv2D(filters=nb_filters_2, kernel_size=[3,3], strides=[1,1], padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters=nb_filters_3, kernel_size=[1,1], strides=[1,1], padding='same')(X)
    X = BatchNormalization()(X)
    
    if res_conv2d:
        X_res = Conv2D(filters=nb_filters_3, kernel_size=[1,1], strides=strides_1, padding='same')(X_start)
        X_res = BatchNormalization()(X_res)
    else:
        X_res = X_start
        
    X = Add()([X, X_res])
    X = Activation('relu')(X)
    return X



def resnet50(input_shape, nb_classes):
    assert len(input_shape) == 3
    
    X_input = Input(shape=input_shape)


    # conv1
    X = Conv2D(filters=64, kernel_size=[7,7], strides=[2,2], padding='same')(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D([3,3], strides=[2,2])(X)

    # conv2_x
    X = residual_block(X, filters=[64, 64, 256], reduce=False, res_conv2d=True)
    X = residual_block(X, filters=[64, 64, 256])
    X = residual_block(X, filters=[64, 64, 256])

    # conv3_x
    X = residual_block(X, filters=[128, 128, 512] ,reduce=True, res_conv2d=True)
    X = residual_block(X, filters=[128, 128, 512])
    X = residual_block(X, filters=[128, 128, 512])
    X = residual_block(X, filters=[128, 128, 512])

    # conv4_x
    X = residual_block(X, filters=[256, 256, 1024], reduce=True, res_conv2d=True)
    X = residual_block(X, filters=[256, 256, 1024])
    X = residual_block(X, filters=[256, 256, 1024])
    X = residual_block(X, filters=[256, 256, 1024])
    X = residual_block(X, filters=[256, 256, 1024])
    X = residual_block(X, filters=[256, 256, 1024])

    # conv5_x
    X = residual_block(X, filters=[512, 512, 2048],  reduce=True, res_conv2d=True)
    X = residual_block(X, filters=[512, 512, 2048])
    X = residual_block(X, filters=[512, 512, 2048])

    X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    X = Dense(units=nb_classes, activation='softmax')(X)
    
      
    model = tf.keras.models.Model(inputs=X_input, outputs=X)  


    return model



def getBatch(array, i, batch_size):
    start_id = i*batch_size
    t = (i+1) * batch_size
    end_id = min( (i+1) * batch_size, array.shape[0])
    batch_array = array[start_id:end_id]
    return batch_array



    
input_pxl=[32, 32,4]
input_obj=[32, 32,8]

model1 = resnet50(input_shape=input_pxl, nb_classes=11)
model2 = resnet50(input_shape=input_obj, nb_classes=11)



mergemodel = Add()([model1.output,model2.output])

mergemodel = Dense(256, activation='relu')(mergemodel)
mergemodel = BatchNormalization()(mergemodel)
# output layer
mergemodel = Dense(units=nb_classes, activation='softmax')(mergemodel)


from keras.models import Model

model = Model([model1.input,model2.input], mergemodel)
"""
model.summary() 
fig=tf.keras.utils.plot_model(model,show_shapes=True)
fig.savefig(fig)
"""

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),   loss="sparse_categorical_crossentropy", metrics=['accuracy'])


loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


#create callbaks
checkpointpath = ('/mnt/DATA/AZZA/3_pixel_object_2D/results/reunion/split_2') # path to sve model
callbacks = [tfk.callbacks.ModelCheckpoint(
              checkpointpath,
              verbose=1, # niveau de log
              monitor='val_accuracy', # metric name
              save_best_only=True, # save only best model
              save_weights_only=True)] # save only weigths


BATCH_SIZE = 34
EPOCHS = 100
hist = model.fit ([x_train,o_train], train_y_enc, validation_data=([x_valid,o_valid],valid_y_enc), batch_size=BATCH_SIZE, epochs=EPOCHS)



valid_pred = model.predict([x_valid,o_valid],batch_size=BATCH_SIZE)


valid_pred = np.argmax(valid_pred,axis=1)
f1_score(valid_y_enc,valid_pred,average='weighted')

test_pred = model.predict([x_test,o_test],batch_size=BATCH_SIZE)
test_pred = encoder.inverse_transform( np.argmax(test_pred,axis=1) )

print ('Acc:',accuracy_score(test_y,test_pred))
print ('F1:',f1_score(test_y,test_pred,average='weighted'))
print ('F1:',f1_score(test_y,test_pred,average=None))
print ('Kappa:',cohen_kappa_score(test_y,test_pred))
# confusion matrix
matrix = confusion_matrix(test_y,test_pred)
print(matrix)
np.save('/mnt/DATA/AZZA/3_pixel_object_2D/results/reunion/split1.npy',test_pred)

labels= [' 0','  1 ','  2 ','  3','  4',' 5',' 6 ','  7','  8',' 9',' 10 ']


cm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
df_cm = DataFrame(cm, index=labels, columns=labels)
ax = sns.heatmap(df_cm,fmt='.2%', annot=False, cmap="YlGnBu")
matrix = ax.get_figure()    
matrix.savefig('/mnt/DATA/AZZA/3_pixel_object_2D/results/reunion/split1.png', dpi=800)



