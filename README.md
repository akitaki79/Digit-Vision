# MNIST Handwritten Digit Recognition App

A web application that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Users can draw digits directly in the browser and get real-time predictions.

## Overview

This project implements a machine learning model for recognizing handwritten digits (0-9) and deploys it as an interactive web application. The model is trained on the MNIST dataset, which contains 70,000 grayscale images of handwritten digits.

The project combines:
- **Deep Learning**: A CNN model built with TensorFlow/Keras
- **Web Development**: An interactive UI created with Streamlit
- **Computer Vision**: Image processing techniques for digit recognition

## Features

- **Interactive Drawing Canvas**: Draw digits using mouse or touchscreen
- **Real-time Prediction**: Instantly see what digit the model recognizes
- **Confidence Visualization**: View confidence levels for each possible digit
- **Tabbed Interface**: Separate tabs for drawing/prediction and model information
- **Responsive Design**: Works on desktop and mobile devices
- **Model Caching**: Train once, predict many times

## Model Architecture

The digit recognition model uses a Convolutional Neural Network (CNN) with the following architecture:

1. **Input Layer**: 28×28×1 (grayscale images)
2. **Convolutional Layers**:
   - Conv2D (32 filters, 3×3 kernel, ReLU activation)
   - MaxPooling (2×2)
   - Conv2D (64 filters, 3×3 kernel, ReLU activation)
   - MaxPooling (2×2)
3. **Flattening**: Convert 2D feature maps to 1D feature vector
4. **Dense Layers**:
   - Dense (128 units, ReLU activation)
   - Dropout (20% for regularization)
   - Dense (10 units, softmax activation)

The model is trained using:
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-entropy
- **Metrics**: Accuracy
- **Epochs**: 5 (configurable)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/akitaki79/Digit-Vision.git
cd mnist-digit-recognition
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the App

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

### Using the App

1. **Draw a Digit**: Use your mouse or touchscreen to draw a digit (0-9) on the canvas
2. **View Prediction**: The model will automatically predict which digit you drew
3. **See Confidence Levels**: The bar chart shows the model's confidence for each possible digit
4. **Clear Canvas**: Click the "Clear Canvas" button to start over
5. **View Model Details**: Switch to the "Model Details" tab for information about the neural network

## Performance

The model typically achieves:
- **Training Accuracy**: ~99.5%
- **Test Accuracy**: ~99%
- **Inference Time**: <100ms per prediction

Performance may vary depending on the hardware and the clarity of the drawn digits.

## Future Improvements

1. **Model Enhancements**:
   - Experiment with different architectures (ResNet, MobileNet)
   - Add support for transfer learning

2. **Application Features**:
   - Add ability to save and load drawings
   - Implement batch recognition for multiple digits
   - Create a more sophisticated drawing interface with undo/redo
   - Add support for recognizing sequences of digits
    
3. **Extended Functionality**:
   - Expand to recognize handwritten letters (A-Z), mathematical symbols and equations, and more

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## License

This project is licensed under the MIT License - see the LICENSE file for details.
