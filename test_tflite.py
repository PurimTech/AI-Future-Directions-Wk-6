import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(img_path):
    img = Image.open(img_path).resize((128,128))
    img = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# Test image
img_path = "test_images/sample.jpg"
input_data = preprocess(img_path)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print("Prediction:", output)
