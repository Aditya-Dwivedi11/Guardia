Face Recognition System – CCTV Ready (Masked \& Unmasked Detection)



📌 Overview



This project is a deep learning–based face recognition system that identifies people from images, live webcam, or CCTV streams.

It works even with very few training images per person and supports:



* Masked faces 😷



* Normal faces 🙂



* Unknown person detection



* Confidence score for every prediction



The system builds face embeddings using a pre-trained deep metric learning model and compares them to known user features stored in a database.



✨ Key Features



* Low-data training – works even with 5–10 images per person



* Mask-compatible recognition



* Real-time CCTV/Webcam inference



* Automatic detection of known vs unknown faces



* Shows confidence score on screen



* Modular and easy to extend



🧠 System Architecture

Input Image / CCTV Frame

&nbsp;       │

Face Detection (MTCNN)

&nbsp;       │

Face Cropping \& Alignment

&nbsp;       │

Face Embedding (InceptionResnetV1)

&nbsp;       │

Compare with stored user embeddings

&nbsp;       │

Display Name + Confidence



📂 Dataset Format



Store all training images like this:



face\_dataset/

│

├── aditya/

│   ├── mask/

│   ├── unmask/

│

├── astha/

│   ├── mask/

│   ├── unmask/

│

├── anshika/

│   ├── mask/

│   ├── unmask/

│

└── avnish/

&nbsp;   ├── mask/

&nbsp;   ├── unmask/





Each folder represents one identity.



🔧 Technologies Used



Python



PyTorch



Facenet (InceptionResnetV1)



MTCNN



OpenCV



NumPy



Google Colab / Jupyter Notebook



🚀 How It Works



1️⃣ Build the Embeddings Database



Every image is converted into a 512-dimensional embedding vector.



Multiple images per user are stored.



All embeddings are saved in a pickle file for fast loading.



2️⃣ Inference



For every detected face:



distance = L2(embedding – stored\_embedding)





If the minimum distance is below a threshold → known user



Otherwise → “Unknown”



A confidence score is calculated:



confidence = max(0, 1 - distance)



✔ Output Example



On detection, each face is shown with:



Name: Astha

Confidence: 93%





If the face is not matched:



Unknown (43%)





Bounding boxes and text are drawn on the output frame.



📌 Where This Works Best



This system performs best when:



Faces are well-lit and clear



CCTV angle is frontal or near-frontal



The user has at least 5–10 images



The face is not extremely occluded



Camera resolution is 480p or higher



Even with masks, the system performs well because embeddings capture deeper features, not only mouth-nose region.



🧪 Ideal Use Cases



Office attendance systems



Hostel / campus entry verification



Home/CCTV monitoring



Face-based unlocking systems



Small AI/ML projects and demos



⚡ Future Improvements



Train a classifier (SVM / Logistic) for faster matching



Face quality scoring



Light normalization



Database expansion



On-device optimized model (TensorRT / OpenVINO)



📁 Files

File	                 Description

build\_embeddings.py	Creates feature embeddings and saves them

face\_embeddings.pkl	Stored database of embeddings

recognition.py        	Live/video recognition

requirements.txt	Dependencies

💡 Author



Astha Tiwari

Computer Science – AI \& ML

PSIT Kanpur

