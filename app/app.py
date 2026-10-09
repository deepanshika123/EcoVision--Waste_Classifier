import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

IMG_SIZE = (224, 224)
THRESHOLD = 0.8

CLASS_NAMES = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

RECYCLABLE = {
    "cardboard": True,
    "glass": True,
    "metal": True,
    "paper": True,
    "plastic": True,
    "trash": False
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f4f8f5;
}

[data-testid="stSidebar"] {
    background: #102f25;
}

[data-testid="stSidebar"] * {
    color: #f1f8f4 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #345447;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #153e2d;
    margin-bottom: 4px;
}

.subtitle {
    color: #65766c;
    font-size: 15px;
    margin-bottom: 24px;
}

.hero {
    background: linear-gradient(120deg, #174c36, #278451);
    padding: 28px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h2 {
    color: white;
    margin: 0 0 8px 0;
    font-size: 26px;
}

.hero p {
    color: #e1f3e7;
    margin: 0;
    line-height: 1.7;
}

.section-title {
    color: #183e2e;
    font-size: 21px;
    font-weight: 700;
    margin: 8px 0 14px 0;
}

.result-card {
    background: white;
    border: 1px solid #dce9df;
    border-radius: 16px;
    padding: 23px;
    margin-top: 10px;
    box-shadow: 0 4px 16px rgba(24, 62, 46, 0.05);
}

.result-label {
    color: #718176;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-value {
    color: #174c36;
    font-size: 29px;
    font-weight: 800;
    margin: 8px 0;
    text-transform: capitalize;
}

.result-detail {
    color: #586b60;
    font-size: 14px;
}

.metric-card {
    background: white;
    border: 1px solid #dce9df;
    border-radius: 14px;
    padding: 18px;
    min-height: 100px;
}

.metric-label {
    color: #718176;
    font-size: 13px;
}

.metric-value {
    color: #174c36;
    font-size: 23px;
    font-weight: 800;
    margin-top: 8px;
}

div[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #8ab99a;
    border-radius: 14px;
    padding: 14px;
}

.stButton button {
    background: #1d7549;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
}

.stButton button:hover {
    background: #145637;
    color: white;
    border: none;
}

.footer {
    color: #75847a;
    text-align: center;
    font-size: 12px;
    padding: 25px 0 5px 0;
}

/* Fix uploader and camera text visibility */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section * {
    color: #183e2e !important;
}

[data-testid="stFileUploader"] button,
[data-testid="stCameraInput"] button {
    background-color: #1d7549 !important;
    color: #ffffff !important;
    border: none !important;
}

[data-testid="stCameraInput"],
[data-testid="stCameraInput"] * {
    color: #183e2e !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    border: 2px dashed #72a985 !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #183e2e !important;
}

[data-testid="stCameraInput"] video {
    border-radius: 12px;
}


/* Fix Upload Image / Use Webcam radio options */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
    color: #183e2e !important;
}

/* Radio button circle */
div[data-testid="stRadio"] label > div:first-child {
    border-color: #1d7549 !important;
}

/* Selected radio option */
div[data-testid="stRadio"] label:has(input:checked) {
    color: #1d7549 !important;
}

/* Option container */
div[data-testid="stRadio"] > div {
    gap: 12px;
}

/* Improve visibility on light background */
div[data-testid="stRadio"] {
    background-color: #e8f2eb;
    padding: 12px 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return keras.models.load_model("models/mobilenetv2.keras")


def predict(image, model):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32")[np.newaxis, ...]
    probs = model.predict(arr, verbose=0)[0]
    return probs

with st.sidebar:
    st.markdown("## ♻️ EcoVision")
    st.caption("SMART WASTE INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["🏠 Dashboard"],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### Waste Categories")
    st.markdown("📦 Cardboard")
    st.markdown("🍾 Glass")
    st.markdown("🔩 Metal")
    st.markdown("📄 Paper")
    st.markdown("🧴 Plastic")
    st.markdown("🗑️ Trash")

   

if page == "ℹ️ About Project":
    st.markdown(
        '<div class="main-title">About EcoVision</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Smart Waste Classifier with Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero">
        <h2>Building a cleaner, smarter future 🌱</h2>
        <p>
        An image classification application that uses a trained
        MobileNetV2 deep learning model to identify waste categories
        and help users understand their recyclability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project Highlights")

    a, b = st.columns(2)

    with a:
        st.markdown("""
        <div class="result-card">
        <h3>🧠 AI Classification</h3>
        Identifies six categories of waste using a trained
        image classification model.
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="result-card">
        <h3>♻️ Recycling Awareness</h3>
        Displays a basic recyclability indicator for the
        predicted category.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Technology Stack")
    st.write(
        "Python · TensorFlow/Keras · MobileNetV2 · "
        "NumPy · Pillow · Streamlit"
    )

    st.info(
        "Note: Actual recyclability depends on local waste "
        "management rules and the condition of the material."
    )

# ---------------- DASHBOARD ----------------
else:
    st.markdown(
        '<div class="main-title">Smart Waste Classifier</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">AI-powered waste recognition and recycling awareness</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero">
        <h2>Give your waste a second chance. 🌿</h2>
        <p>
        Upload a photo or use your camera to identify waste.
        Get an AI prediction, confidence score and recycling guidance
        in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top metrics
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">WASTE CATEGORIES</div>
        <div class="metric-value">06</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">MODEL</div>
        <div class="metric-value">MobileNetV2</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">CONFIDENCE THRESHOLD</div>
        <div class="metric-value">80%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns([1, 1.15], gap="large")

    image = None

    # Input section
    with left:
        st.markdown(
            '<div class="section-title">📷 Upload Waste Image</div>',
            unsafe_allow_html=True
        )

        mode = st.radio(
            "Choose image source",
            ["Upload Image", "Use Webcam"],
            horizontal=True
        )

        if mode == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose JPG, JPEG or PNG",
                type=["jpg", "jpeg", "png"]
            )

            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                except Exception:
                    st.error("Could not open this image. Please try another.")

        else:
            camera_file = st.camera_input("Take a photo")

            if camera_file is not None:
                try:
                    image = Image.open(camera_file)
                except Exception:
                    st.error("Could not open this photo. Please try again.")

        if image is None:
            st.info(
                "Upload an image or capture a photo to begin classification."
            )
        else:
            st.image(
                image,
                caption="Selected waste image",
                use_container_width=True
            )

    # Results section
    with right:
        st.markdown(
            '<div class="section-title">🔍 Classification Results</div>',
            unsafe_allow_html=True
        )

        if image is not None:
            try:
                with st.spinner("Analyzing your image..."):
                    model = load_model()
                    probs = predict(image, model)

                if len(probs) != len(CLASS_NAMES):
                    st.error(
                        "Model output does not match the six configured classes."
                    )
                else:
                    top = int(np.argmax(probs))
                    label = CLASS_NAMES[top]
                    conf = float(probs[top])
                    recyclable = RECYCLABLE[label]

                    status = "Recyclable" if recyclable else "Non-recyclable"
                    symbol = "♻️" if recyclable else "🗑️"

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">PREDICTED CATEGORY</div>
                        <div class="result-value">{label}</div>
                        <div class="result-detail">
                            Model confidence: <b>{conf:.1%}</b>
                        </div>
                        <hr style="border:0;border-top:1px solid #e4ece6;margin:18px 0;">
                        <div class="result-label">RECYCLING INDICATOR</div>
                        <div style="font-size:21px;font-weight:700;color:#174c36;margin-top:8px;">
                            {symbol} {status}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if conf < THRESHOLD:
                        st.warning(
                            f"Low confidence: the model predicts '{label}', "
                            "but the result may be unreliable. Please verify manually."
                        )
                    else:
                        st.success("Prediction confidence meets the 80% threshold.")

                    st.markdown("#### Category Probabilities")

                    chart_data = {
                        CLASS_NAMES[i]: float(probs[i])
                        for i in range(len(CLASS_NAMES))
                    }

                    st.bar_chart(chart_data, horizontal=True)

                    st.caption(
                        "Probabilities are model scores. Correct classification "
                        "depends on the model's training and calibration."
                    )

            except Exception as e:
                st.error(f"Could not run classification: {e}")

        else:
            st.markdown("""
            <div class="result-card">
                <div class="result-label">YOUR RESULTS WILL APPEAR HERE</div>
                <div class="result-value">Ready to analyze</div>
                <div class="result-detail">
                    Your predicted category, confidence and recycling
                    indicator will appear after you provide an image.
                </div>
            </div>
            """, unsafe_allow_html=True)

   