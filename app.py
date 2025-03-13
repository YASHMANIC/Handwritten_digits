from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
from tensorflow import keras
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random
app = Flask(__name__)

cd = os.getcwd()
model = tf.keras.models.load_model(f'{cd}/Handwritten-digits.model.keras')

# Load the MNIST dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5,validation_split=0.2,
    batch_size=32)

# Function to evaluate model performance
def evaluate_model():
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    print(classification_report(y_test, y_pred))

# Evaluate the model
evaluate_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = request.form['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    # Convert to image
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Resize to 28x28 and convert to array
    image = image.resize((28, 28))
    image_array = np.array(image)
    
    # Normalize and reshape
    image_array = image_array.reshape(1,28, 28) 
    
    # Save the processed image with random number
    random_num = random.randint(1, 10000)
    plt.imsave(f'prediction_{random_num}.png', image_array[0], cmap='gray')

    # Make prediction
    prediction = model.predict(image_array)
    digit = np.argmax(prediction[0])

    return jsonify({'digit': int(digit), 'confidence': float(prediction[0][digit])})

if __name__ == '__main__':
    # Only use Flask's development server in development mode
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(debug=True)
    else:
        # Production mode - let Gunicorn handle it
        app.run(host='0.0.0.0', port=5000)