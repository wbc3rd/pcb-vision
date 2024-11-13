import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import numpy as np

# Define custom labels
LABELS = {
    0: "Bad Capacitor",
    1: "Bad IC",
    2: "Bad Resistor",
    3: "Good Capacitor",
    4: "Good IC",
    5: "Good Resistor"
}

# Load model
model = torch.load("./model.pth")
model.eval()

# Define inference function
def inference(img, model, threshold=0.5):
    # Ensure model is in evaluation mode
    model.eval()

    # Convert image to RGB
    img = img.convert("RGB")

    # Transform image to tensor
    transform = T.Compose([
        T.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        pred = model(img_tensor)  # Forward pass

    # Convert image back to PIL for drawing bounding boxes
    transform = T.ToPILImage()
    img = transform(img_tensor.squeeze(0))  # Remove batch dimension

    # Get predicted bounding boxes, labels, and scores
    boxes = pred[0]["boxes"].cpu().detach().numpy()
    labels = pred[0]["labels"].cpu().detach().numpy()
    scores = pred[0]["scores"].cpu().detach().numpy()

    # Filter predictions based on a confidence threshold
    boxes = [box for box, score in zip(boxes, scores) if score > threshold]
    labels = [label for label, score in zip(labels, scores) if score > threshold]
    scores = [score for score in scores if score > threshold]

    # Draw bounding boxes, labels, and scores on the image
    draw = ImageDraw.Draw(img)
    
    # Load custom font
    try:
        font = ImageFont.truetype("arial.ttf", size=50) 
    except IOError:
        font = ImageFont.load_default() 
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="#ff0000cc", width=3)

        # Get custom label name from LABELS dictionary
        label_name = LABELS.get(label, "Unknown Label")
        
        # Draw label and score text
        text = f"{label_name}, Score: {score:.2f}"
        
        # Calculate text size and position
        text_bbox = draw.textbbox((x1, y1 - 10), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Draw background for the text
        draw.rectangle([x1, y1 - text_height - 5, x1 + text_width, y1], fill="black")

        # Draw the actual text
        draw.text((x1, y1 - text_height - 5), text, fill="white", font=font)

    return img, pred

# Streamlit UI
st.title("PCB Component Detection with Custom Faster R-CNN")
st.write("Upload an image and run the component detection model")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file)
    
    # Run inference
    result_img, predictions = inference(img, model)
    
    # Display result
    st.image(result_img, caption="Processed Image", use_container_width=True)
    
    # Display prediction results (e.g., bounding boxes, labels, and scores)
    #st.write("Predicted Bounding Boxes, Labels, and Scores:")
    #for idx, (box, label, score) in enumerate(zip(predictions[0]["boxes"].cpu().detach().numpy(), 
    #                                               predictions[0]["labels"].cpu().detach().numpy(),
    #                                               predictions[0]["scores"].cpu().detach().numpy())):
    #    if score > 0.5:  # Only show predictions with high confidence
    #        label_name = LABELS.get(label, "Unknown Label")  # Use custom label names
    #        st.write(f"Box {idx+1}: {box}, Label: {label_name}, Score: {score:.2f}")
