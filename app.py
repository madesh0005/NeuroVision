import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from deepface import DeepFace
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import tempfile
import av
from collections import Counter

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="NeuroVision - Emotion AI",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1526378722370-44d1c7d0f3a4");
    background-size: cover;
}
.main-title {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: white;
}
.sub-title {
    text-align: center;
    color: #f0f2f6;
    font-size: 20px;
}
.block-container {
    background: rgba(0,0,0,0.75);
    padding: 2rem;
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 NeuroVision</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Professional Emotion Intelligence System</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- MODE SELECTION ---------------- #
mode = st.sidebar.selectbox(
    "Select Mode",
    ["📷 Image Analysis", "🎥 Video Analysis", "🔴 Real-Time Camera"]
)

# =========================================================
# 📷 IMAGE ANALYSIS
# =========================================================
if mode == "📷 Image Analysis":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing Emotion..."):
            result = DeepFace.analyze(
                image_np,
                actions=['emotion'],
                enforce_detection=False
            )

        emotions = result[0]['emotion']
        dominant = result[0]['dominant_emotion']

        st.success(f"Dominant Emotion: {dominant.upper()}")

        df = pd.DataFrame(emotions.items(), columns=["Emotion", "Confidence"])
        fig = px.bar(df, x="Emotion", y="Confidence", color="Emotion")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 🎥 VIDEO ANALYSIS
# =========================================================
elif mode == "🎥 Video Analysis":

    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

    if video_file is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        emotion_list = []
        frame_count = 0

        st.info("Processing video... Please wait.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 10th frame for efficiency
            if frame_count % 10 == 0:
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        enforce_detection=False
                    )
                    emotion = result[0]['dominant_emotion']
                    emotion_list.append(emotion)
                except:
                    pass

            frame_count += 1

        cap.release()

        if emotion_list:
            emotion_counts = Counter(emotion_list)
            dominant_video_emotion = emotion_counts.most_common(1)[0][0]

            st.success(f"Overall Dominant Emotion in Video: {dominant_video_emotion.upper()}")

            df = pd.DataFrame(emotion_counts.items(), columns=["Emotion", "Frequency"])
            fig = px.pie(df, names="Emotion", values="Frequency", title="Emotion Distribution")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("No face detected in video.")

# =========================================================
# 🔴 REAL-TIME CAMERA
# =========================================================
elif mode == "🔴 Real-Time Camera":

    st.info("Real-time emotion detection started.")

    emotion_tracker = []

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            try:
                result = DeepFace.analyze(
                    img,
                    actions=['emotion'],
                    enforce_detection=False
                )

                emotion = result[0]['dominant_emotion']
                emotion_tracker.append(emotion)

                cv2.putText(
                    img,
                    emotion.upper(),
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            except:
                pass

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=VideoProcessor
    )

    if st.button("Generate Session Summary"):
        if emotion_tracker:
            counts = Counter(emotion_tracker)
            dominant_live = counts.most_common(1)[0][0]

            st.success(f"Session Dominant Emotion: {dominant_live.upper()}")

            df = pd.DataFrame(counts.items(), columns=["Emotion", "Frequency"])
            fig = px.bar(df, x="Emotion", y="Frequency", color="Emotion")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No emotions recorded yet.")
