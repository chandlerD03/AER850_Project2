import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf

model = tf.keras.models.load_model('my_model.h5')  

model_input_shape = model.input_shape[1:3]  

class_names = ['Crack', 'Missing.Head', 'Paint-off']  

image_paths = [
    r"Data\test\crack\test_crack.jpg",  
    r"Data\test\missing-head\test_missinghead.jpg", 
    r"Data\test\paint-off\test_paintoff.jpg" 
]

def process_and_predict_image(img_path, model, class_names, img_size=model_input_shape):
    actual_class = os.path.basename(os.path.dirname(img_path))

    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0  
    

    predictions = model.predict(img_array)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx]

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Actual Class: {actual_class} | Predicted Class: {predicted_class} ({confidence * 100:.1f}% confidence)", pad=20)
    
    probability_text = "\n".join([f"{class_names[i]}: {predictions[i] * 100:.1f}%" for i in range(len(class_names))])
    plt.text(0.5, -0.1, probability_text, ha='center', va='top', transform=plt.gca().transAxes, fontsize=12, color='green')
    plt.show()

# Loop through the specified images and predict
for img_path in image_paths:
    print(f"Processing image: {img_path}")
    process_and_predict_image(img_path, model, class_names)





 