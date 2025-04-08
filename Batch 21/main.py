import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw

# Initialize the Inference HTTP Client with your API details
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="f7kWJaMjaV59ujyAkdoD"
)

def draw_bounding_box(image, predictions):
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(image)

    for prediction in predictions:
        # Extract center coordinates, dimensions, and class details
        x_center = prediction['x']
        y_center = prediction['y']
        width = prediction['width']
        height = prediction['height']
        confidence = prediction['confidence']
        label = prediction['class']

        # Convert center coordinates to top-left and bottom-right coordinates
        x1 = x_center - (width / 2)
        y1 = y_center - (height / 2)
        x2 = x_center + (width / 2)
        y2 = y_center + (height / 2)

        # Draw the bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Prepare text for label and confidence
        text = f"{label}: {confidence:.2f}"

        # Draw the label above the bounding box
        text_x = x1
        text_y = y1 - 10  # Position the text above the box
        draw.text((text_x, text_y), text, fill="red")

    return image

def main():
    st.title("Oil Spill Detection Using SAR Images")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and resize the uploaded image
        image = Image.open(uploaded_file)
        image = image.resize((640, 360))  # Resize to 640x360 pixels

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Perform inference
        if st.button("Detect Oil Spill"):
            # Save the uploaded file temporarily
            temp_image_path = "temp_image.jpg"
            image.save(temp_image_path)

            # Call the inference client
            result = CLIENT.infer(temp_image_path, model_id="-biv3n/1")

            # Check if 'predictions' exist in result
            if 'predictions' in result:
                predictions = result['predictions']
                
                if predictions:
                    # Pass the original image to the drawing function
                    detected_image = draw_bounding_box(image.copy(), predictions)

                    # Display the image with bounding boxes
                    st.image(detected_image, caption='Detected Oil Spills', use_container_width=True)
                else:
                    st.warning("No oil spills detected in the image.")
            else:
                st.error("Failed to retrieve predictions from the detection API.")

if __name__ == "__main__":
    main()
