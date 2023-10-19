import os, glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report

file_path = "image"
name_class = os.listdir(file_path)
print(name_class)

filepaths = list(glob.glob(file_path+'/**/*.*'))
print(filepaths)

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
print(labels)

filepath = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')
data = pd.concat([filepath, labels], axis=1)
data = data.sample(frac=1).reset_index(drop=True)
#print(data.head(5))

counts = data.Label.value_counts()
sns.barplot(x=counts.index, y=counts)
plt.xlabel('Type')
#plt.xticks(rotation=90)
#plt.show()

train, test = train_test_split(data, test_size=0.25, random_state=42)
fig, axes = plt.subplots(nrows=3, figsize=(10,8), subplot_kw={'xticks':[],'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(data.Filepath[i]))
    ax.set_title(data.Label[i])
#plt.tight_layout()
#plt.show()

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='Filepath',
    y_col='Label',
    target_size=(100,100),
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)
valid_gen = train_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(100,100),
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed=42
)
test_gen = test_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(100,100),
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed=42
)

pretrained_model = ResNet50(
    input_shape=(100,100,3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
pretrained_model.trainable = False

inputs = pretrained_model.input

x = Dense(128, activation='relu')(pretrained_model.output)
x = Dense(128, activation='relu')(x)
outputs = Dense(40, activation='softmax')(x)             #Important!
model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

my_callbacks = [EarlyStopping(monitor='val_accuracy',
                            min_delta=0,
                            patience=2,
                            mode='auto')]

history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=3
)

model.save("model_resnet.h5")

pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
plt.title("Accuracy")
plt.show()

pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title("Loss")
plt.show()

results = model.evaluate(test_gen, verbose=0)
print("Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

pred = model.predict(test_gen)
pred = np.argmax(pred, axis=1)
labels = (train_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]
y_test = list(test.Label)
#print(classification_report(y_test, pred))

#fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12,8),
#                        subplot_kw={'xticks': [], 'yticks': []})
#for i, ax in enumerate(axes.flat):
#    ax.imshow(plt.imread(test.Filepath.iloc[i]))
#    ax.set_title(f"True: {test.Label.iloc[i]}\nPerdicted: {pred[i]}")
#plt.tight_layout()
#plt.show()

import cv2
import numpy as np
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input

img_path = 'demo2.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (100,100))
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)
result = model.predict(x)
#print((result*100).astype('int'))
#plt.imshow(img)

p = list((result*100).astype('int'))
pp = list(p[0])
#print(pp)
#print("Largest element is:", max(pp))
index = pp.index(max(pp))
name_class = ['Aloevera', 'Tulsi', 'Ashwagandha', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Avacado', 'Bamboo', 'Basale', 'Betel', 'Betel_Nut', 'Brahmi', 'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka', 'Ganike', 'Gauva', 'Geranium', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon', 'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 'Pappaya', 'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Wood_sorel']
#print(name_class[index])
# Print the accuracy
print(f"This is {name_class[index]} ({max(pp)}% Accuracy) ")

#plt.title(name_class[index])
#plt.imshow(img)