# import streamlit as st
# import numpy as np
# import pydicom
# from tensorflow.keras.models import load_model
# from skimage.transform import resize
# import matplotlib.pyplot as plt
# import pandas as pd
# from datetime import datetime

# # Set page config
# st.set_page_config(
#     page_title="Medical Image Analysis",
#     page_icon="🏥",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS to beautify the app
# st.markdown("""
#     <style>
#         .main > div {
#             padding: 2rem;
#         }
#         .stButton>button {
#             width: 100%;
#             margin-top: 1rem;
#         }
#         .reportview-container {
#             background: #f0f2f6
#         }
#         .css-1d391kg {
#             padding-top: 3.5rem;
#         }
#         .stProgress > div > div > div > div {
#             background-color: #09ab3b;
#         }
#         .cancer-progress > div > div > div > div {
#             background-color: #ff4b4b;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Function to process DICOM image
# def process_dicom_image(dicom_file):
#     ds = pydicom.dcmread(dicom_file)
#     pixel_array = ds.pixel_array
#     processed_image = resize(pixel_array, (256, 256, 1), anti_aliasing=True)
#     return ds, pixel_array, processed_image

# # Function to make prediction
# def predict_cancer(model, processed_image):
#     input_image = processed_image.reshape(1, 256, 256, 1)
#     prediction = model.predict(input_image)
#     probability = prediction[0][1] * 100
#     is_cancer = np.round(prediction[0])[1] == 1
#     return probability, is_cancer

# # Function to extract DICOM metadata
# def get_dicom_metadata(ds):
#     try:
#         metadata = {
#             "Patient Name": str(ds.PatientName) if hasattr(ds, 'PatientName') else "N/A",
#             "Patient ID": ds.PatientID if hasattr(ds, 'PatientID') else "N/A",
#             "Patient Age": ds.PatientAge if hasattr(ds, 'PatientAge') else "N/A",
#             "Patient Sex": ds.PatientSex if hasattr(ds, 'PatientSex') else "N/A",
#             "Study Date": datetime.strptime(ds.StudyDate, "%Y%m%d").strftime("%Y-%m-%d") if hasattr(ds, 'StudyDate') else "N/A",
#             "Modality": ds.Modality if hasattr(ds, 'Modality') else "N/A",
#             "Body Part": ds.BodyPartExamined if hasattr(ds, 'BodyPartExamined') else "N/A",
#             "Image Size": f"{ds.Rows}x{ds.Columns}" if hasattr(ds, 'Rows') and hasattr(ds, 'Columns') else "N/A",
#             "Manufacturer": ds.Manufacturer if hasattr(ds, 'Manufacturer') else "N/A",
#             "Institution": ds.InstitutionName if hasattr(ds, 'InstitutionName') else "N/A"
#         }
#         return metadata
#     except Exception as e:
#         st.error(f"Error extracting metadata: {str(e)}")
#         return {}

# def main():
#     # Sidebar
#     with st.sidebar:
#         st.image("3053987_0.webp", 
#                  use_container_width=True)
#         st.title("Navigation")
#         page = st.radio("Go to", ["Home", "About", "Help"])
        
#         st.markdown("---")
#         st.markdown("### Model Information")
#         st.info("""
#         This application uses a CNN model trained on medical images to detect cancer.
#         - Input size: 256x256
#         - Model type: Sequential CNN
#         - Output: Binary classification
#         """)
        
#         st.markdown("---")
#         st.markdown("### Developer Info")
#         st.markdown("Created by: Your Name")
#         st.markdown("[GitHub Repository](https://github.com)")

#     # Main content
#     if page == "Home":
#         st.title("🏥 Medical Image Cancer Classification")
#         st.write("""
#         This application analyzes medical DICOM images to assist in cancer detection.
#         Upload a DICOM (.dcm) file to get started.
#         """)
        
#         # Load model
#         try:
#             with st.spinner("Loading model..."):
#                 model = load_model('model_dicom_cancer.h5')
#             st.success("✅ Model loaded successfully!")
#         except Exception as e:
#             st.error(f"Error loading model: {str(e)}")
#             return
        
#         # File uploader with custom styling
#         uploaded_file = st.file_uploader(
#             "Choose a DICOM file", 
#             type=['dcm'],
#             help="Upload a DICOM (.dcm) file for analysis"
#         )
        
#         if uploaded_file is not None:
#             try:
#                 # Process the image
#                 with st.spinner("Processing image..."):
#                     ds, original_image, processed_image = process_dicom_image(uploaded_file)
                
#                 # Create three columns for layout
#                 col1, col2 = st.columns([2, 2])
                
#                 # Column 1: Original Image
#                 with col1:
#                     st.markdown("### Original DICOM Image")
#                     fig, ax = plt.subplots(figsize=(10, 10))
#                     ax.imshow(original_image, cmap="bone")
#                     ax.axis('off')
#                     st.pyplot(fig)
                
#                 # Column 2: DICOM Information
#                 with col2:
#                     st.markdown("### DICOM Metadata")
#                     metadata = get_dicom_metadata(ds)
                    
#                     # Create a styled DataFrame for metadata
#                     df = pd.DataFrame(list(metadata.items()), columns=['Attribute', 'Value'])
#                     st.dataframe(df.set_index('Attribute'), use_container_width=True)
                
#                 # Column 3: Analysis Results
#                 # with col3:
#                 st.markdown("### Analysis Results")
                
#                 # Make prediction
#                 with st.spinner("Analyzing..."):
#                     probability, is_cancer = predict_cancer(model, processed_image)
                
#                 # Display prediction with custom styling
#                 st.metric(
#                     label="Cancer Probability",
#                     value=f"{probability:.2f}%",
#                     delta="High Risk" if probability > 50 else "Low Risk"
#                 )
                
#                 # Custom progress bar based on prediction
#                 if is_cancer:
#                     st.markdown('<div class="cancer-progress">', unsafe_allow_html=True)
#                     st.progress(probability / 100)
#                     st.markdown('</div>', unsafe_allow_html=True)
#                     st.error(f"⚠️ Cancer detected")
#                 else:
#                     st.progress(probability / 100)
#                     st.success(f"✅ No cancer detected")
            
#                 # Additional information and disclaimer
#                 st.markdown("---")
#                 st.markdown("### Important Notice")
#                 st.warning("""
#                 **Medical Disclaimer**: This AI-assisted analysis is for reference only and should not be used as the sole basis for medical decisions. 
#                 Please consult with qualified healthcare professionals for proper diagnosis and treatment.
#                 """)
                
#             except Exception as e:
#                 st.error(f"Error processing image: {str(e)}")
#                 st.write("Please ensure you've uploaded a valid DICOM file.")

#     elif page == "About":
#         st.title("About This Application")
#         st.write("""
#         ### Purpose
#         This application is designed to assist medical professionals in the preliminary screening of cancer using DICOM images.
        
#         ### Technology Stack
#         - Streamlit for the web interface
#         - TensorFlow for the deep learning model
#         - PyDICOM for medical image processing
#         - Scikit-image for image preprocessing
        
#         ### Model Architecture
#         The underlying model is a Convolutional Neural Network (CNN) trained on a dataset of medical images.
#         """)

#     else:  # Help page
#         st.title("Help & Documentation")
#         st.write("""
#         ### How to Use This App
#         1. Navigate to the Home page
#         2. Upload a DICOM (.dcm) file using the file uploader
#         3. Wait for the analysis to complete
#         4. Review the results and metadata
        
#         ### Troubleshooting
#         - Ensure your DICOM file is properly formatted
#         - Check that the file size is within acceptable limits
#         - Verify that the image is of sufficient quality
        
#         ### Contact Support
#         For technical support or questions, please contact:
#         - Email: support@example.com
#         - Phone: (555) 123-4567
#         """)

# if __name__ == "__main__":
#     main()


import streamlit as st
import numpy as np
import pydicom
from tensorflow.keras.models import load_model
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (keeping your existing styles)
st.markdown("""
    <style>
        .main > div {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .reportview-container {
            background: #f0f2f6
        }
        .css-1d391kg {
            padding-top: 3.5rem;
        }
        .stProgress > div > div > div > div {
            background-color: #09ab3b;
        }
        .cancer-progress > div > div > div > div {
            background-color: #ff4b4b;
        }
    </style>
""", unsafe_allow_html=True)

# Function to load patient mapping data
def load_patient_data():
    try:
        df = pd.read_csv('dicom_data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading patient mapping data: {str(e)}")
        return None

# Function to find file by patient ID
def find_file_by_patient_id(patient_id, patient_data, folder_path):
    if patient_data is not None:
        patient_row = patient_data[patient_data['Patient ID'] == patient_id]
        if not patient_row.empty:
            filename = patient_row.iloc[0]['Filename']
            file_path = os.path.join(folder_path, filename)
            if os.path.exists(file_path):
                return file_path, patient_row.iloc[0]
    return None, None 

# Your existing functions
def process_dicom_image(dicom_file):
    ds = pydicom.dcmread(dicom_file)
    pixel_array = ds.pixel_array
    processed_image = resize(pixel_array, (256, 256, 1), anti_aliasing=True)
    return ds, pixel_array, processed_image

def predict_cancer(model, processed_image):
    input_image = processed_image.reshape(1, 256, 256, 1)
    prediction = model.predict(input_image)
    probability = prediction[0][1] * 100
    is_cancer = np.round(prediction[0])[1] == 1
    return probability, is_cancer

def get_dicom_metadata(ds):
    try:
        metadata = {
            "Patient Name": str(ds.PatientName) if hasattr(ds, 'PatientName') else "N/A",
            "Patient ID": ds.PatientID if hasattr(ds, 'PatientID') else "N/A",
            "Patient Age": ds.PatientAge if hasattr(ds, 'PatientAge') else "N/A",
            "Patient Sex": ds.PatientSex if hasattr(ds, 'PatientSex') else "N/A",
            "Study Date": datetime.strptime(ds.StudyDate, "%Y%m%d").strftime("%Y-%m-%d") if hasattr(ds, 'StudyDate') else "N/A",
            "Modality": ds.Modality if hasattr(ds, 'Modality') else "N/A",
            "Body Part": ds.BodyPartExamined if hasattr(ds, 'BodyPartExamined') else "N/A",
            "Image Size": f"{ds.Rows}x{ds.Columns}" if hasattr(ds, 'Rows') and hasattr(ds, 'Columns') else "N/A",
            "Manufacturer": ds.Manufacturer if hasattr(ds, 'Manufacturer') else "N/A",
            "Institution": ds.InstitutionName if hasattr(ds, 'InstitutionName') else "N/A"
        }
        return metadata
    except Exception as e:
        st.error(f"Error extracting metadata: {str(e)}")
        return {}

def main():
    # Sidebar
    with st.sidebar:
        st.image("3053987_0.webp", use_container_width=True)
        st.title("Navigation")
        page = st.radio("Go to", ["Home", "About", "Help"])
        
        st.markdown("---")
        st.markdown("### Model Information")
        st.info("""
        This application uses a CNN model trained on medical images to detect cancer.
        - Input size: 256x256
        - Model type: Sequential CNN
        - Output: Binary classification
        """)
        
        st.markdown("---")
        st.markdown("### Developer Info")
        st.markdown("Created by: Dharaa,Dharshini,Sarvika")
        st.markdown("[GitHub Repository](https://github.com)")

    # Main content
    if page == "Home":
        st.title("🏥 Medical Image Cancer Classification")
        st.write("""
        This application analyzes medical DICOM images to assist in cancer detection.
        You can either upload a DICOM file directly or search by ID.
        """)
        
        # Load model
        try:
            with st.spinner("Loading model..."):
                model = load_model('model_dicom_cancer.h5')
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
        
        # Load patient data
        patient_data = load_patient_data()
        
        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["Upload File", "Search by ID"])
        
        with tab1:
            # Original file uploader functionality
            uploaded_file = st.file_uploader(
                "Choose a DICOM file", 
                type=['dcm'],
                help="Upload a DICOM (.dcm) file for analysis"
            )
            file_to_process = uploaded_file
            
        with tab2:
            if patient_data is not None:
                dc_id = st.text_input("Enter Doctor ID")
                if dc_id:
                    # Patient ID search functionality
                    patient_ids = patient_data['Patient ID'].tolist()
                    selected_id = st.selectbox(
                        "Select Patient ID",
                        options=patient_ids,
                        help="Select a patient ID to load their DICOM file"
                    )
                    
                    if selected_id:
                        file_path, patient_info = find_file_by_patient_id(
                            selected_id, 
                            patient_data, 
                            "Dataset"  # Replace with your actual folder path
                        )
                        
                        if file_path:
                            st.success(f"Found file for patient {selected_id}")
                            file_to_process = file_path
                        else:
                            st.error("File not found for the selected Patient ID")
                            file_to_process = None
            else:
                st.error("Patient mapping data not available")
                file_to_process = None
        
        # Process the file (whether uploaded or found by ID)
        if file_to_process is not None:
            try:
                # Process the image
                with st.spinner("Processing image..."):
                    ds, original_image, processed_image = process_dicom_image(file_to_process)
                
                # Create columns for layout
                col1, col2 = st.columns([2, 2])
                
                # Column 1: Original Image
                with col1:
                    st.markdown("### Original DICOM Image")
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(original_image, cmap="bone")
                    ax.axis('off')
                    st.pyplot(fig)
                
                # Column 2: DICOM Information
                with col2:
                    st.markdown("### DICOM Metadata")
                    metadata = get_dicom_metadata(ds)
                    df = pd.DataFrame(list(metadata.items()), columns=['Attribute', 'Value'])
                    st.dataframe(df.set_index('Attribute'), use_container_width=True)
                
                # Analysis Results
                st.markdown("### Analysis Results")
                
                # Make prediction
                with st.spinner("Analyzing..."):
                    probability, is_cancer = predict_cancer(model, processed_image)
                
                # Display prediction
                st.metric(
                    label="Cancer Probability",
                    value=f"{probability:.2f}%",
                    delta="High Risk" if probability > 50 else "Low Risk"
                )
                
                # Progress bar based on prediction
                if is_cancer:
                    st.markdown('<div class="cancer-progress">', unsafe_allow_html=True)
                    st.progress(probability / 100)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.error(f"⚠️ Cancer detected")
                else:
                    st.progress(probability / 100)
                    st.success(f"✅ No cancer detected")
                
                # Disclaimer
                st.markdown("---")
                st.markdown("### Important Notice")
                st.warning("""
                **Medical Disclaimer**: This AI-assisted analysis is for reference only and should not be used as the sole basis for medical decisions. 
                Please consult with qualified healthcare professionals for proper diagnosis and treatment.
                """)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.write("Please ensure you've uploaded a valid DICOM file.")

    # Keep your existing About and Help pages
    elif page == "About":
        st.title("About This Application")
        st.write("""
        ### Purpose
        This application is designed to assist medical professionals in the preliminary screening of cancer using DICOM images.
        
        ### Technology Stack
        - Streamlit for the web interface
        - TensorFlow for the deep learning model
        - PyDICOM for medical image processing
        - Scikit-image for image preprocessing
        
        ### Model Architecture
        The underlying model is a Convolutional Neural Network (CNN) trained on a dataset of medical images.
        """)

    else:  # Help page
        st.title("Help & Documentation")
        st.write("""
        ### How to Use This App
        1. Navigate to the Home page
        2. Choose either:
           - Upload a DICOM (.dcm) file directly
           - Search for a patient by their ID
        3. Wait for the analysis to complete
        4. Review the results and metadata
        
        ### Troubleshooting
        - Ensure your DICOM file is properly formatted
        - Check that the file size is within acceptable limits
        - Verify that the image is of sufficient quality
        
        ### Contact Support
        For technical support or questions, please contact:
        - Email: support@example.com
        - Phone: (555) 123-4567
        """)

if __name__ == "__main__":
    main()