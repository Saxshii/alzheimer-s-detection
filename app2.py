import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ------------------- Page Configuration -------------------

st.set_page_config(page_title="Alzheimer Detection", page_icon="🧠", layout="wide")

# ------------------- Load Model -------------------

model = tf.keras.models.load_model('my_model.h5')
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# ------------------- Preprocess Function -------------------

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((176, 176))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# ------------------- Custom CSS -------------------

st.markdown("""
<style>
body { background-color: #ffffff; }
h1, h2, h3, h4, h5, h6 { color: #003366; font-family: 'Helvetica'; }
.sidebar .sidebar-content { background-color: #f5f5f5; }
.stButton>button {
    background-color: #0052cc;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    border: none;
}
.stButton>button:hover {
    background-color: #003d99;
}
.uploadedImage {
    border-radius: 10px;
    border: 2px solid #0052cc;
    padding: 5px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Sidebar Navigation -------------------

st.sidebar.title("🧠 Alzheimer Detection")
page = st.sidebar.radio("Navigate", ["Home", "Scan MRI"])

# ------------------- HOME PAGE -------------------

if page == "Home":
    st.title("🧠 Alzheimer's Disease Detection")
    st.subheader("Early Diagnosis • Better Care • Improved Quality of Life")

    
    st.markdown("""
<style>
.info-card {
    padding: 18px;
    background: rgba(255,255,255,0.06);
    border-left: 6px solid #4da3ff;
    border-radius: 12px;
    margin-bottom: 20px;
    font-size: 17px;
    color: #dce7f7;
}

/* Progression Stage Boxes */
.stage-box {
    background: #111827; /* Soft dark card to match black theme */
    border: 1px solid #2f3b4d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 3px 8px rgba(0,0,0,0.25);
    transition: 0.25s ease;
    color: #e5eeff;
    font-weight: 500;
}
.stage-box:hover {
    transform: translateY(-6px);
    border-color: #4da3ff;
    box-shadow: 0 6px 18px rgba(77,163,255,0.35);
}

/* Make headings look more premium */
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 650;
    color: #e8f1ff !important;
}

/* Paragraph text */
p {
    color: #d0dae8 !important;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


    # Intro box
    st.markdown("""
    <div class="info-card">
    Alzheimer's disease slowly damages brain cells, affecting memory, thinking, and behavior.
    Early detection through MRI scans allows better medical planning and improved outcomes.
    </div>
    """, unsafe_allow_html=True)

    # Stages section
    st.markdown("### **Alzheimer’s Progression Stages**")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown("""<div class="stage-box"><h4>Mild</h4><p>Memory lapses, slight confusion.</p></div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="stage-box"><h4>Moderate</h4><p> noticeable personality and behavior changes.</p></div>""", unsafe_allow_html=True)
    c3.markdown("""<div class="stage-box"><h4>Very Mild</h4><p>Small cognitive changes only visible in testing.</p></div>""", unsafe_allow_html=True)
    c4.markdown("""<div class="stage-box"><h4>Non-Demented</h4><p>Healthy cognition and memory.</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### **MRI Brain Scan Examples**")

    col1, col2, col3 = st.columns(3)
    with col1: st.image("images/sample1.jpg", caption="Healthy Brain", use_container_width=True)
    with col2: st.image("images/sample2.jpg", caption="Mild Dementia", use_container_width=True)
    with col3: st.image("images/sample3.jpg", caption="Moderate Dementia", use_container_width=True)

    st.markdown("---")
    st.markdown("### **Why Early Detection Matters**")
    st.write("""
    - 🩺 Helps plan treatment early  
    - 🧠 Slows decline through therapy & lifestyle  
    - 🤝 Supports family preparedness  
    - 💙 Preserves independence longer  
    """)

    st.markdown("<br><br><p style='text-align:center; color:#666;'>Developed for academic research and awareness.</p>", unsafe_allow_html=True)



# ------------------- SCAN PAGE -------------------

if page == "Scan MRI":
    st.title("🧠 Alzheimer Detection Scan")
    st.write("Upload your MRI image to let our trained deep learning model analyze and predict the Alzheimer's stage.")
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", width=350)
        if st.button("Analyze Scan"):
            with st.spinner("Analyzing..."):
               img_array = preprocess_image(image)
               prediction = model.predict(img_array)
               predicted_class = class_labels[np.argmax(prediction)]
               confidence = (np.max(prediction ) * 100) 

            st.success(f"### 🧬 Prediction: {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")


