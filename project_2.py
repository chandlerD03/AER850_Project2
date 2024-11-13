import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.models import Sequential


#Step 1

img_shape = (100, 100, 3)

train_dir = r"Data\train"  
val_dir = r"Data\valid"   

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,    
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True     
)



val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)




print(f"Training data batch shape: {train_generator.image_shape}")
print(f"Number of training batches: {len(train_generator)}")
print(f"Validation data batch shape: {val_generator.image_shape}")
print(f"Number of validation batches: {len(val_generator)}")



#step 2+3

model = Sequential()

model.add(Conv2D(512, (3, 3), activation='relu', strides=(1, 1), input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 2
model.add(Conv2D(256, (3, 3), activation='relu',strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5)) 

model.add(Dense(3, activation='softmax')) 


model.summary()

#Step 4

import matplotlib.pyplot as plt


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  
    verbose=1
)

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()


plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()

model.save('my_model.h5') 
