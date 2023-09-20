import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet

# Load the pre-trained MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False)

# Add a global average pooling layer and a new classification layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # You can adjust the number of neurons
predictions = Dense(num_classes, activation='softmax')(x)  # num_classes is the number of plant species

# Create the final model
model = keras.Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using your dataset
model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size)
