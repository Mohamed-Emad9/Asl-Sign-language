ASL Sign Language Recognition
University Deep Learning Project

 1- Project Description 
This project presents a Deep Learning–based system for recognizing American Sign Language (ASL) letters and words from images and video frame sequences.
The system combines Convolutional Neural Networks (CNNs) for letter recognition with Transformer-based models for word recognition, all integrated into a simple and user-friendly Streamlit graphical interface with optional Text-to-Speech support.

2 -Project Objectives
Recognize ASL alphabet letters from single images
Recognize ASL words from sequences of frames
Apply modern Deep Learning techniques (CNNs and Transformers)
Provide an intuitive graphical user interface
Enhance accessibility using audio output  

3-  System Overview
The system consists of two main recognition pipelines:
Letters Recognition
Model: CNN (ResNet18)
Input: Single image
Output: Predicted ASL letter
Words Recognition
Model: CNN + Transformer Encoder
Input: Sequence of frames
Output: Predicted ASL word
The application is deployed using Streamlit, allowing users to upload images or capture snapshots from a camera and receive predictions both visually and audibly.

4- Key Features
ResNet18-based CNN for accurate letter classification
Transformer-based architecture for word-level sequence recognition
Robust preprocessing and data augmentation
Complete training pipeline (Training, Validation, Testing)
Clean and minimal Streamlit GUI
Camera snapshot integration
Text-to-Speech output
Automatic class loading using ImageFolder directory structure

 5- Project Structure
project/
├── models/
│   ├── letters_model.pth
│   └── words_model.pth
├── train/
│   ├── train_letters.py
│   └── train_words.py
├── utils/
│   ├── preprocess.py
│   └── model_loader.py
├── app.py
├── requirements.txt
└── README.md

6- Team Members & Responsibilities
Name
Responsibility
Mohamed Emad Elraw
Data preprocessing and augmentation
Zeyad Ahraf
CNN model architecture (Letters)
Sarah Mahrous Mohamed
Model training
Shahd Ahmed Khaled
Validation and testing
Hana Alaa Abderhman
Transformer model (Words recognition)
Youssif Hisham Baiomy
Text-to-Speech and project documentation
Ahd Said Atia
Camera integration
Omar Abdelhameed
GUI design and model integration
Requirements
Install the required dependencies using:

pip install -r requirements.txt
Main Libraries
streamlit
torch
torchvision
numpy
opencv-python
pillow
tqdm
pyttsx3

 7- How to Run the Project
1. Dataset Preparation
The dataset should follow the ImageFolder format:

dataset/
├── train/
│   ├── class1/
│   ├── class2/
└── test/
    ├── class1/
    ├── class2/
2. Model Training (Optional)

python train/train_letters.py
python train/train_words.py
Trained models will be saved in the models/ directory.
3. Run the Application
streamlit run app.py  

8- Training and Evaluation Summary
Letters Model: ResNet18, CrossEntropyLoss, Adam optimizer
Words Model: CNN combined with Transformer Encoder
Evaluation Metrics: Accuracy, Precision, Recall, Confusion Matrix
Overfitting Control: Data augmentation and validation-based checkpointing

9-Future Improvements

Real-time ASL recognition from live video
Sentence-level sign language recognition
Expansion to larger and more diverse datasets
Deployment as a mobile or web-based application 

10- Conclusion
This project demonstrates a practical and accessible application of Deep Learning, Computer Vision, and Transformer models for American Sign Language recognition, providing both visual and audio-based feedback through a user-friendly interface.