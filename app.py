import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Handwritten Digit Recognition")

def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

st.title("✏️ Handwritten Digit Recognition")
st.markdown("""
Draw a digit (0-9) in the canvas below and the model will predict which digit it is.
""")

tab1, tab2 = st.tabs(["Draw & Predict", "Model Details"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
    
    with col2:
        st.write("### Prediction")
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            img_gray = np.mean(img[:, :, :3], axis=2)  
            img_resized = tf.image.resize(img_gray[..., np.newaxis], [28, 28]).numpy() 

            img_normalized = img_resized / 255.0
            
            @st.cache_resource
            def get_model():

                model = create_model()
                
                # train model
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
                y_train = to_categorical(y_train, 10)
                model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=0)
                return model
            
            model = get_model()
            
            prediction = model.predict(img_normalized[np.newaxis, ...])
            predicted_digit = np.argmax(prediction)
            
            # prediction and confidence
            st.markdown(f"## Predicted Digit: {predicted_digit}")
            
            # Plot confidence levels
            fig, ax = plt.subplots(figsize=(4, 2))
            confidence_values = prediction[0] * 100
            ax.bar(range(10), confidence_values)
            ax.set_xticks(range(10))
            ax.set_xlabel('Digit')
            ax.set_ylabel('Confidence (%)')
            ax.set_ylim(0, 100)
            st.pyplot(fig)
            
        if st.button("Clear Canvas"):
            st.rerun()

with tab2:
    st.header("About the Model")
    st.write("""
    This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits.
    
    **Model Architecture:**
    - 2 Convolutional layers (32 and 64 filters)
    - Max pooling after each conv layer
    - 128-unit dense layer with ReLU activation
    - Dropout layer (20%) for regularization
    - Output layer with softmax activation
    
    **Training:**
    - Trained on 60,000 MNIST training images
    - Typically achieves ~99% accuracy on test set
    """)
    
    st.subheader("How to use:")
    st.markdown("""
    1. Draw a digit from 0-9 in the canvas
    2. The model will automatically predict which digit you drew
    3. The bar chart shows the model's confidence for each possible digit
    4. Click "Clear Canvas" to try again
    """)
    
    st.info("Note: For best results, draw the digit clearly and centered in the canvas.")