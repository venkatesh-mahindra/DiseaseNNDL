import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import time
import base64
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .prediction-box {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .symptom-selector {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Cache functions for better performance
@st.cache_resource
def load_model():
    """Load the trained model from .h5 file"""
    try:
        # Load model from .h5 file
        model = tf.keras.models.load_model('disease_prediction_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Make sure 'disease_prediction_model.h5' file exists in the current directory.")
        return None

@st.cache_resource
def load_label_encoder():
    """Load the label encoder from pickle file"""
    try:
        # Load label encoder from pickle file
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return label_encoder
    except Exception as e:
        st.error(f"Error loading label encoder: {e}")
        st.error("Make sure 'label_encoder.pkl' file exists in the current directory.")
        return None

@st.cache_data
def load_data():
    """Load the dataset and get symptom columns"""
    try:
        # Load the dataset
        df = pd.read_csv("extracted_files/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
        
        # Get all symptom columns (all columns except 'diseases')
        symptom_cols = [col for col in df.columns if col != 'diseases']
        
        return df, symptom_cols
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Function to create radar chart for symptoms
def create_radar_chart(selected_symptoms):
    categories = selected_symptoms
    values = [1] * len(selected_symptoms)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Selected Symptoms',
        line_color='rgb(31, 119, 180)',
        fillcolor='rgba(31, 119, 180, 0.5)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Symptom Pattern",
        height=400
    )
    
    return fig

# Function to create bar chart for prediction probabilities
def create_prediction_chart(probabilities, class_names, top_n=5):
    # Get top N predictions
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    top_probs = probabilities[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    # Create a DataFrame for the chart
    chart_data = pd.DataFrame({
        'Disease': top_classes,
        'Probability': top_probs
    })
    
    # Create the bar chart
    fig = px.bar(
        chart_data, 
        x='Probability', 
        y='Disease',
        orientation='h',
        color='Probability',
        color_continuous_scale='Viridis',
        title=f"Top {top_n} Predicted Diseases",
        labels={'Probability': 'Confidence Score', 'Disease': ''},
        height=400
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_range=[0, 1]
    )
    
    return fig

# Function to get disease information
def get_disease_info(disease_name):
    # This is a placeholder. In a real app, you would have a database or API call
    # to get detailed information about each disease.
    disease_info = {
        "Common Cold": {
            "description": "A viral infectious disease of the upper respiratory tract that primarily affects the nose.",
            "treatment": "Rest, fluids, over-the-counter medications for symptoms",
            "prevention": "Regular handwashing, avoiding close contact with sick individuals"
        },
        "Pneumonia": {
            "description": "Infection that inflames air sacs in one or both lungs, which may fill with fluid.",
            "treatment": "Antibiotics for bacterial pneumonia, antiviral medications for viral pneumonia",
            "prevention": "Vaccination, good hygiene practices"
        },
        "Diabetes": {
            "description": "A group of metabolic disorders characterized by high blood sugar levels over a prolonged period.",
            "treatment": "Insulin therapy, oral medications, lifestyle changes",
            "prevention": "Healthy diet, regular exercise, maintaining a healthy weight"
        }
    }
    
    # Return default info if disease not in our database
    default_info = {
        "description": "A medical condition with specific symptoms and treatments.",
        "treatment": "Consult with a healthcare professional for proper diagnosis and treatment.",
        "prevention": "Regular check-ups and following medical advice can help prevent complications."
    }
    
    return disease_info.get(disease_name, default_info)

# Main app function
def main():
    # App title and introduction
    st.title("🏥 Disease Prediction System")
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #1565c0;">Welcome to the Disease Prediction System</h3>
        <p>This application uses machine learning to predict possible diseases based on your symptoms. 
        Please select the symptoms you're experiencing, and our system will analyze them to provide potential diagnoses.</p>
        <p><strong>Disclaimer:</strong> This tool is for informational purposes only and should not replace professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and label encoder
    model = load_model()
    label_encoder = load_label_encoder()
    
    # Load data
    df, symptom_cols = load_data()
    
    if df is None or model is None or label_encoder is None:
        st.error("Failed to load necessary components. Please check the error messages above.")
        st.info("""
        Make sure the following files exist in your directory:
        - disease_prediction_model.h5 (TensorFlow model file)
        - label_encoder.pkl (Pickle file containing the label encoder)
        - extracted_files/Final_Augmented_dataset_Diseases_and_Symptoms.csv (Dataset file)
        """)
        return
    
    # Get all unique diseases for reference
    all_diseases = sorted(df['diseases'].unique())
    
    # Create two columns for the layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="symptom-selector">', unsafe_allow_html=True)
        st.subheader("Select Your Symptoms")
        
        # Search box for symptoms
        search_term = st.text_input("Search for symptoms", "")
        
        # Filter symptoms based on search term
        filtered_symptoms = symptom_cols if not search_term else [
            symptom for symptom in symptom_cols 
            if search_term.lower() in symptom.lower()
        ]
        
        # Show number of symptoms found
        st.write(f"Found {len(filtered_symptoms)} symptoms matching your search")
        
        # Create a multiselect for symptoms with a scrollable area
        selected_symptoms = st.multiselect(
            "Select all that apply:",
            options=filtered_symptoms,
            default=[]
        )
        
        # Add a "Clear All" button
        if st.button("Clear All Selections"):
            selected_symptoms = []
            st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show the radar chart for selected symptoms
        if selected_symptoms:
            radar_chart = create_radar_chart(selected_symptoms)
            st.plotly_chart(radar_chart, use_container_width=True)
    
    with col2:
        st.subheader("Prediction Results")
        
        # Create a placeholder for the prediction
        prediction_placeholder = st.empty()
        
        # Predict button
        predict_button = st.button("Predict Disease", key="predict_button")
        
        if predict_button:
            if not selected_symptoms:
                st.warning("Please select at least one symptom before predicting.")
            else:
                # Show a spinner while processing
                with st.spinner("Analyzing symptoms..."):
                    # Create input features (all 0s initially)
                    input_features = pd.DataFrame(0, index=[0], columns=symptom_cols)
                    
                    # Set selected symptoms to 1
                    for symptom in selected_symptoms:
                        if symptom in input_features.columns:
                            input_features[symptom] = 1
                    
                    # Make prediction
                    prediction = model.predict(input_features)[0]
                    
                    # Get the predicted class and probability
                    predicted_class_index = np.argmax(prediction)
                    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
                    confidence = prediction[predicted_class_index]
                    
                    # Get disease information
                    disease_info = get_disease_info(predicted_class)
                    
                    # Display prediction results
                    with prediction_placeholder.container():
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        
                        # Display the main prediction with confidence
                        st.markdown(f"""
                        <h2 style='text-align: center; color: #1976d2;'>Predicted Disease</h2>
                        <h1 style='text-align: center; margin-bottom: 20px;'>{predicted_class}</h1>
                        <p style='text-align: center; font-size: 18px;'>Confidence: <b>{confidence:.2%}</b></p>
                        <hr>
                        """, unsafe_allow_html=True)
                        
                        # Display disease information
                        st.markdown("<h3>About this condition:</h3>", unsafe_allow_html=True)
                        st.write(disease_info["description"])
                        
                        st.markdown("<h3>Common treatments:</h3>", unsafe_allow_html=True)
                        st.write(disease_info["treatment"])
                        
                        st.markdown("<h3>Prevention:</h3>", unsafe_allow_html=True)
                        st.write(disease_info["prevention"])
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Display the prediction chart
                        st.plotly_chart(
                            create_prediction_chart(prediction, label_encoder.classes_, top_n=5),
                            use_container_width=True
                        )
                        
                        # Disclaimer
                        st.info("""
                        **Medical Disclaimer**: This prediction is based on the symptoms you selected and should not be 
                        considered as a definitive diagnosis. Please consult with a healthcare professional for proper 
                        medical advice and treatment.
                        """)
    
    # Add a section for disease information
    st.markdown("""
    <hr>
    <h2 style='text-align: center; margin-top: 30px;'>Disease Information</h2>
    """, unsafe_allow_html=True)
    
    # Create three columns for disease categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px; height: 100%;'>
            <h3 style='color: #2e7d32;'>Respiratory Diseases</h3>
            <ul>
                <li>Common Cold</li>
                <li>Pneumonia</li>
                <li>Asthma</li>
                <li>Bronchitis</li>
                <li>Tuberculosis</li>
            </ul>
            <p>Respiratory diseases affect the lungs and other parts of the respiratory system.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #fff3e0; padding: 15px; border-radius: 10px; height: 100%;'>
            <h3 style='color: #e65100;'>Cardiovascular Diseases</h3>
            <ul>
                <li>Hypertension</li>
                <li>Heart Attack</li>
                <li>Stroke</li>
                <li>Arrhythmia</li>
                <li>Heart Failure</li>
            </ul>
            <p>Cardiovascular diseases involve the heart and blood vessels.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; height: 100%;'>
            <h3 style='color: #0d47a1;'>Metabolic Disorders</h3>
            <ul>
                <li>Diabetes</li>
                <li>Hypothyroidism</li>
                <li>Hyperthyroidism</li>
                <li>Obesity</li>
                <li>Gout</li>
            </ul>
            <p>Metabolic disorders affect the body's ability to process certain nutrients and chemicals.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <hr>
    <p style='text-align: center; color: #666; padding: 20px;'>
        Disease Prediction System | Created with Streamlit | © 2023
    </p>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
