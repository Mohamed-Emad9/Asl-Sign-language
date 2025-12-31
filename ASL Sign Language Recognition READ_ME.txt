ASL Sign Language Recognition
University Deep Learning Project

1- Project Description
This project presents a Deep Learning–based system for recognizing American Sign Language (ASL) letters and words from images and video frame sequences. The system combines Convolutional Neural Networks (CNNs) for letter recognition with Transformer-based models for word recognition, all integrated into a simple and user-friendly Streamlit graphical interface with optional Text-to-Speech support. 

2- Project Objectives
The main objective of this project is to recognize ASL alphabet letters from single images and ASL words from sequences of video frames. The project applies modern Deep Learning techniques using CNNs and Transformers, provides an intuitive graphical user interface, and enhances accessibility through optional audio output.


3- System Overview
The system consists of two main recognition pipelines. The letter recognition pipeline uses a CNN based on ResNet18 to classify single input images and output the predicted ASL letter. The word recognition pipeline combines a CNN with a Transformer Encoder to process sequences of frames and output the predicted ASL word. The entire system is deployed using Streamlit, allowing users to upload images or capture camera snapshots and receive predictions both visually and audibly.


4- Key Features
The project utilizes a ResNet18-based CNN for accurate letter classification and a Transformer-based architecture for word-level sequence recognition. It includes robust preprocessing and data augmentation techniques, a complete training pipeline covering training, validation, and testing phases, a clean and minimal Streamlit graphical interface, camera snapshot integration, Text-to-Speech output, and automatic class loading using the ImageFolder directory structure.


5- Project Structure
The project follows a modular directory structure where trained models are stored in the models folder, training scripts are placed in the train directory, and shared utilities such as preprocessing and model loading are organized under utils. The main application logic and graphical user interface are implemented in app.py, while dependency management and project documentation are handled through requirements.txt and README.md.

6- Team Members & Responsibilities
Mohamed Emad Elraw – Data preprocessing and augmentation
Zeyad Ahraf – CNN model architecture (Letters)
Sarah Mahrous Mohamed – Model training
Shahd Ahmed Khaled – Validation and testing
Hana Alaa Abderhman – Transformer model (Words recognition)
Youssif Hisham Baiomy – Text-to-Speech and project documentation
Ahd Said Atia – Camera integration
Omar Abdelhameed – GUI design and model integration
Requirements
The required dependencies can be installed using the provided requirements file. The main libraries used in this project include Streamlit, PyTorch, Torchvision, NumPy, OpenCV, Pillow, TQDM, and pyttsx3.

7- How to Run the Project
The dataset should be prepared following the ImageFolder directory structure, where each class is placed in a separate folder for training and testing. Model training is optional since pre-trained models are provided, but the training scripts can be executed if retraining is needed. After installing the required dependencies, the application can be launched using Streamlit, enabling users to upload images or capture camera snapshots and receive predictions with optional audio output.

8- Training and Evaluation Summary
The letter recognition model is based on ResNet18 and trained using CrossEntropyLoss with the Adam optimizer. The word recognition model combines a CNN with a Transformer Encoder. Model performance is evaluated using accuracy, precision, recall, and confusion matrices, while overfitting is controlled through data augmentation and validation-based checkpointing.

9- Future Improvements
Future enhancements include real-time ASL recognition from live video, sentence-level sign language recognition, expansion to larger and more diverse datasets, and deployment as a mobile or web-based application.

10- Conclusion
This project demonstrates a practical and accessible application of Deep Learning, Computer Vision, and Transformer models for American Sign Language recognition. By integrating visual predictions with optional audio feedback through a user-friendly interface, the system improves accessibility and usability for a wide range of users.
