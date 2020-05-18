
from keras.callbacks import LearningRateScheduler
from keras.models import model_from_json
from cnn.resnet import ResNet
from sklearn.metrics import classification_report
import numpy as np
import argparse
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import operator
from keras.optimizers import SGD

# set the matplotlib backend so figures can be saved in the background
import matplotlib

# define the # of epochs, initial learning rate and batch size
num_epochs = 2
init_lr= 1e-1
bs = 32
 
# create a function called polynomial decay which helps us decay our 
# learning rate after each epoch

def poly_decay(epoch):
  # initialize the maximum # of epochs, base learning rate,
  # and power of the polynomial
  maxEpochs = num_epochs
  baseLR = init_lr
  power = 1.0  # turns our polynomial decay into a linear decay
 
  # compute the new learning rate based on polynomial decay
  alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
 
  # return the new learning rate
  return alpha



# Step 2 - Preparing the train/test data and training the model

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=10,
                                                 color_mode='rgb',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=10,
                                            color_mode='rgb',
                                            class_mode='categorical')
                                            

model = ResNet.build(64, 64, 3, 32, (3, 4),
  (64, 128, 256), reg=0.0005)
opt = SGD(lr=init_lr, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
  metrics=["accuracy"])

# define our set of callbacks and fit the model
callbacks = [LearningRateScheduler(poly_decay)]

history = model.fit_generator(
        training_set,
        steps_per_epoch=1757, # No of images in training set
        epochs=2,
        validation_data=test_set,
        validation_steps=478,
        callbacks=callbacks)# No of images in test set



# Saving the model
model_json = model.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model-bw.h5')

image_predict = image.load_img('data/test/c/hand4_c_bot_seg_1_cropped.jpeg',target_size=(64, 64),color_mode='rgb')

image_predict = image.img_to_array(image_predict)
image_predict = np.expand_dims(image_predict, axis=0)

result = model.predict(image_predict)
prediction = {    'ZERO': result[0][0],
                  'ONE': result[0][1],
                  'TWO': result[0][2],
                  'THREE': result[0][3],
                  'FOUR': result[0][4],
                  'FIVE': result[0][5],
                  'a': result[0][6],
                  'b': result[0][7],
                  'c': result[0][8],
                  'd': result[0][9],
                  'e': result[0][10],
                  'f': result[0][11],
                  'g': result[0][12],
                  'h': result[0][13],
                  'i': result[0][14],
                  'j': result[0][15],
                  'k': result[0][16],
                  'l': result[0][17],
                  'm': result[0][18],
                  'n': result[0][19],
                  'o': result[0][20],
                  'p': result[0][21],
                  'q': result[0][22],
                  'r': result[0][23],
                  's': result[0][24],
                  't': result[0][25],
                  'u': result[0][26],
                  'v': result[0][27],
                  'w': result[0][28],
                  'x': result[0][29],
                  'y': result[0][30],
                  'z': result[0][31],
              }
# Sorting based on top prediction
prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

print(prediction)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()