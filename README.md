# Enhancing Medical Image Diagnosis Using Convolutional Neural Network and Transfer Learning

### Pneumonia Detection API (CNN Microservice)

This repository contains the Python-based machine learning microservice designed to integrate with a broader MERN stack application. It serves as the inference engine for analyzing chest X-ray images and predicting the presence of pneumonia using a trained DenseNet model.

## 🚀 Features

* **RESTful Inference API**: Built with FastAPI to serve predictions efficiently to a frontend or Node.js/Express backend.
* **Dual Input Methods**:
* **File Uploads**: Supports direct image uploads via `FormData` (`/predict-file` endpoint).
* **URL Processing**: Asynchronously fetches and processes images from provided URLs (`/predict-url` endpoint).


* **Automated Data Splitting**: Includes a utility script to automatically partition the `chest_xray` dataset into training (70%), validation (15%), and testing (15%) sets for model training.
* **Asynchronous Processing**: Offloads heavy model prediction tasks to separate threads to ensure the API remains non-blocking and responsive.

## 🛠️ Tech Stack

* **Web Framework**: FastAPI, Uvicorn, Pydantic
* **Machine Learning & Vision**: TensorFlow (v2.15.0), NumPy, Pillow
* **Data Processing**: `split-folders` for dataset management
* **HTTP Client**: `httpx` (for async image fetching), `requests`

## 📂 Project Structure

```text
.
├── app.py                  # Main FastAPI application and endpoints
├── split_data.py           # Dataset partitioning script (Train/Val/Test)
├── requirements.txt        # Project dependencies
├── model/
│   └── pneumonia_densenet.keras  # Trained DenseNet121 model weights
└── dataset_split/          # Generated directory for structured data

```

## ⚙️ Installation & Setup

1. **Clone the repository** and navigate to the project directory.
2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Install dependencies**:
```bash
pip install -r requirements.txt

```


4. **Prepare the Dataset** (If training a new model):
Update the `input_folder` path in `split_data.py` to point to your raw dataset, then run:
```bash
python split_data.py

```


5. **Start the API Server**:
```bash
python app.py

```


*The server will start on `http://0.0.0.0:8000*`.

## 📡 API Endpoints

### 1. Health Check

* **GET** `/health`
* **Description**: Checks if the Python server is running and verifies that the `.keras` model has been successfully loaded into memory.

### 2. Predict from File

* **POST** `/predict-file`
* **Description**: Accepts an image file and returns a prediction.
* **Payload**: `multipart/form-data` with a file field.
* **Response**:
```json
{
  "prediction": "PNEUMONIA",
  "confidenceScore": 94.50,
  "filename": "scan123.jpg"
}

```



### 3. Predict from URL

* **POST** `/predict-url`
* **Description**: Accepts an image URL, downloads the image asynchronously, and returns a prediction.
* **Payload**: `application/json`
```json
{
  "imageURL": "https://example.com/path/to/xray.jpg"
}

```



## 🧠 Model Details

The microservice utilizes a saved Keras model (`pneumonia_densenet.keras`). Incoming images are dynamically resized to 224x224 pixels and normalized (rescaled by 1/255.0) to match the exact preprocessing steps used during the `ImageDataGenerator` training phase before being passed to the model for inference. Binary classification logic is applied, with a raw score of `>= 0.5` classifying the image as "PNEUMONIA".
