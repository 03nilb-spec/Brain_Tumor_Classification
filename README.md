# Brain_Tumor_Classification
Deep learning-based brain tumor classification using MRI images with Streamlit deployment.

## Live Demo
Try the web app here:  
👉 [Brain Tumor Classification Streamlit App](https://braintumorclassification-7rmjvhrnwtp2g69v78fklc.streamlit.app/)

## Features
- Custom CNN and Pretrained (Transfer Learning) models.
- Model performance comparison.
- Streamlit web app to predict tumor type from uploaded MRI images.
- Clean, modular, and well-documented code.

## Tumor Classes
1. **Glioma**
2. **Meningioma**
3. **Pituitary**
4. **No Tumor**

## Getting Started

### 1. Clone the Repository

git clone https://github.com/03nilb-spec/Brain_Tumor_Classification.git
cd Brain_Tumor_Classification

### 2. Set up Environment 
python -m venv venv
venv\Scripts\activate   

### 3. Install Dependencies
pip install -r requirements.txt

### 4 Run the Streamlit App
streamlit run app.py

## Note 
- Dataset is not included in this repository.
- The model file pretrained_model.h5 is used for inference in the web app.

