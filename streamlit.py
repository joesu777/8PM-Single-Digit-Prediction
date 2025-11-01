import streamlit as st
from fastai.vision.all import *

st.title("Single Digit Prediction")
st.text("Built by Joel Suwanto")
import sys, types
sys.modules['fasttransform'] = types.ModuleType('fasttransform')

def number_label(file_path):
    file_parts = str(file_path).split("/")
    #print(file_parts)
    folder_name = file_parts[-2] # Sample0

    return folder_name[-1]

single_digit_model = load_learner("single_digit_model.pkl")

def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = single_digit_model.predict(img)
    return pred_class

uploaded_file = st.file_uploader("Upload an image of a digit...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    prediction = predict(uploaded_file)
    st.subheader(f"Predicted Digit: {prediction}")


st.text("Built with Streamlit and FastAI")