# app.py
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
from models.model import DeepLearningModel
import joblib
from scripts.xai_eval import convert_to_gradcam
import cv2


# Bucket name
BUCKET_NAME = "aipi540-cv"
VERTEX_AI_ENDPOINT = ""

# class type
class_names = ["Normal", "Mild Diabetic Retinopathy", "Severe Diabetic Retinopathy"]


# need to change the following code
class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3205, 0.2244, 0.1613], 
                               std=[0.2996,0.2158, 0.1711]) ## I think this is out of date slightly? at least with what VGG uses
        ])

    
    # Preprocess the image
    def preprocess_image(self, image):
        """Preprocess the image for model input."""
        img_tensor = self.transform(image)
        return img_tensor.unsqueeze(0).to(self.device)

def load_model(model_type):
    handler = ModelHandler()
    device = handler.device
    
    # Load the model
    model = DeepLearningModel()
    model.load_state_dict(torch.load("models/vgg16_model.pth", map_location=device))
    
    model = model.to(device)
    if hasattr(model, 'eval'):
        model.eval()
    
    return model, handler.preprocess_image

# Prediction function
def predict(model, image_tensor):
    '''Predict the class of the input image using the given model.'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() 

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
    
    #  Convert outputs to probabilities and predicted class
    probabilities = torch.softmax(outputs, dim=1)  
    predicted_class = torch.argmax(probabilities, dim=1).item()
    class_probabilities = probabilities[0].cpu().numpy()

    return predicted_class, class_probabilities

def generate_gradcam(model, image_tensor):
    """Generate Grad-CAM heatmap for the input image using the given model."""
    try:
        cam = convert_to_gradcam(model)
        heatmap = cam(input_tensor=image_tensor, targets=None)  

        # Remove batch dimension and convert to numpy array
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.squeeze().cpu().numpy() 
        else:
            heatmap = heatmap.squeeze()  

        # Normalize the heatmap to [0, 1] range
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        return heatmap
    except Exception as e:
        return f"Grad-CAM Error: {str(e)}"

# the streamlit app
def main():
    st.set_page_config(
        page_title="Diabetic Retinopathy Prediction",
        page_icon="👁️",
        layout="wide"
    )
    
    st.title("👁️ Diabetic Retinopathy Detection System")
    st.write("Upload a fundus image to detect diabetic retinopathy severity")

    # 只加载 Deep Learning Model (VGG16)
    st.sidebar.header("Model Information")
    st.sidebar.write("Using Deep Learning Model (VGG16)")

    # 加载 Deep Learning Model
    model, preprocess = load_model("Deep Learning Model")

    st.sidebar.header("About")
    st.sidebar.markdown("""
    This system aims to detect diabetic retinopathy (DR) from fundus images.
    ### Model:
    - **Deep Learning Model**: VGG16-based architecture
    
    ### Classes:
    - Normal (No DR)
    - Mild DR
    - Severe DR
    """)

    st.header("Image Upload")
    uploaded_file = st.file_uploader(
        "Choose a fundus image", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image"):
            try:
                processed_image = preprocess(image)

                with st.spinner("Analyzing image..."):
                    predicted_class, class_probs = predict(model, processed_image)

                st.success("Analysis Complete!")

                # Display prediction results
                st.header("Prediction Results")
                st.write(f"**Predicted Condition:** {class_names[predicted_class]}")
                st.write("**Class Probabilities:**")
                st.json({class_names[i]: float(class_probs[i]) for i in range(len(class_probs))})

                # *
                with st.spinner("Generating XAI..."):
                    heatmap = generate_gradcam(model, processed_image)

                st.header("Grad-CAM Explanation")
                if isinstance(heatmap, str):  
                    st.error(heatmap)
                else:
                    st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

            

                

if __name__ == "__main__":
    main()