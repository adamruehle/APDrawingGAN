import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Load TFLite model
tflite_model_path = "preprocess/1.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape'][1:3]

def segment_person(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to expected model input size
    img = cv2.resize(img, input_shape)

    # Normalize pixel values
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get the segmentation mask
    mask = interpreter.get_tensor(output_details[0]['index'])[0]

    # Thresholding the mask to get the person segmentation
    mask = np.argmax(mask, axis=-1)
    person_mask = np.where(mask == 15, 255, 0).astype(np.uint8)
    resized_image = cv2.resize(person_mask, (512, 512))

    return resized_image

# Path to your image
image_path = "dataset/data/test_single/DavidWeaver_aligned.png"

# Perform segmentation
segmented_image = segment_person(image_path)

save_path = "dataset/mask/ALL/DavidWeaver_aligned.png"
# Save the segmented image
cv2.imwrite(save_path, segmented_image)