import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array, array_to_img
import os
import tensorflow_addons as tfa

# Paths for both models
pix2pix_model_path = r'C:\Users\narendra\Documents\Streamlit\P2P_gen.h5'
cyclegan_model_path = r'C:\Users\narendra\Documents\Streamlit\Cycle_gen.h5'

# Constants
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Define a class to manage session state
class SessionState:
    def __init__(self):
        self.selected_model = "Pix2Pix"
        self.pix2pix_loaded = False
        self.cyclegan_loaded = False
        self.pix2pix_model = None
        self.cyclegan_model = None

# Create a session state instance
session_state = SessionState()

# Function to save uploaded file
def save_uploaded_file(uploaded_file, folder):
    file_path = os.path.join(folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to preprocess a single image for Pix2Pix model
def preprocess_pix2pix_image(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_image(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])  # Resize image to fixed dimensions
    image = (image * 2) - 1  # Normalize to [-1, 1]
    return image

# Function to preprocess a single image for CycleGAN model
def preprocess_cyclegan_image(image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img = img_to_array(image)
    if img.shape[-1] == 1:  # If grayscale, convert to 3 channels
        img = np.concatenate([img, img, img], axis=-1)
    img = (img / 127.5) - 1  # Normalize to [-1, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to post-process the generated image
def postprocess_image(predicted_img):
    predicted_img = (predicted_img + 1) * 127.5  # Denormalize to [0, 255]
    predicted_img = np.clip(predicted_img, 0, 255)  # Clip values to [0, 255]
    predicted_img = predicted_img.astype('uint8')  # Convert to uint8
    return predicted_img

# Function to load Pix2Pix model
def load_pix2pix_model():
    global session_state
    session_state.pix2pix_model = tf.keras.models.load_model(pix2pix_model_path)
    session_state.pix2pix_loaded = True

# Function to load CycleGAN model
def load_cyclegan_model():
    global session_state
    custom_objects = {'InstanceNormalization': tfa.layers.InstanceNormalization}
    session_state.cyclegan_model = tf.keras.models.load_model(cyclegan_model_path, custom_objects=custom_objects)
    session_state.cyclegan_loaded = True

# Streamlit app
def main():
    st.title("Image Colorization")

    # Dropdown to select model
    session_state.selected_model = st.selectbox("Select Model", ["Pix2Pix", "CycleGAN"])

    # Load the selected model if not already loaded
    if session_state.selected_model == "Pix2Pix":
        if not session_state.pix2pix_loaded:
            load_pix2pix_model()

        uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_path = save_uploaded_file(uploaded_file, "test_input")
            
            # Read and preprocess the uploaded image
            image = preprocess_pix2pix_image(file_path)
            image = tf.expand_dims(image, axis=0)  # Add batch dimension
            
            # Generate colorized image
            prediction = session_state.pix2pix_model(image, training=True)
            
            # Display input and output images
            st.write("## Input Image")
            st.image(uploaded_file, use_column_width=True, channels="GRAY")
            
            st.write("## Colorized Image (Pix2Pix)")
            prediction = prediction[0].numpy() * 0.5 + 0.5  # Denormalize to [0, 1]
            st.image(prediction, use_column_width=True)

            # Cleanup: Remove the uploaded file from the test_input_folder
            os.remove(file_path)

    elif session_state.selected_model == "CycleGAN":
        if not session_state.cyclegan_loaded:
            load_cyclegan_model()

        uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Load the uploaded image
            image = Image.open(uploaded_file).convert("L")  # Convert to grayscale

            # Preprocess the test image
            test_image = preprocess_cyclegan_image(image)

            # Generate the colorized image using the CycleGAN model
            generated_image = session_state.cyclegan_model(test_image, training=False)

            # Post-process the generated image
            generated_image = postprocess_image(generated_image[0])

            # Display the original and generated images
            st.write("## Original Grayscale Image")
            st.image(image, use_column_width=True)

            st.write("## Colorized Image (CycleGAN)")
            st.image(array_to_img(generated_image), use_column_width=True)

if __name__ == '__main__':
    main()
