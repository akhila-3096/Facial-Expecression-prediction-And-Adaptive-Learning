# Main Streamlit dashboard.
# Replace this starter with the full dashboard version from your working project
# if you have additional UI customizations not included in this source package.
import sys
from pathlib import Path
import av
import cv2
import time
import threading
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.face_detection import FaceDetector
from src.emotion_prediction import EmotionPredictor
from database.db_manager import (
    create_session, add_observation, end_session,
    get_sessions, get_observations, get_database_summary
)

st.set_page_config(page_title="NeuroVision AI", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
.stApp {background:#080D1A;color:#F1F5FF;font-family:Inter,sans-serif;}
[data-testid="stSidebar"] {background:#101B32;}
[data-testid="stMetric"] {background:#192945;border:1px solid #30466B;
border-radius:14px;padding:18px;}
.stButton>button,.stDownloadButton>button {background:#557EFF;color:white;
border:0;border-radius:10px;font-weight:700;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return FaceDetector(), EmotionPredictor()

try:
    detector, predictor = load_models()
except Exception as exc:
    st.error(f"Could not load AI models: {exc}")
    st.info("Add the Haar Cascade XML and trained checkpoint as described in README.md.")
    st.stop()

def recommendation(emotion):
    suggestions = {
        "happy": "Continue with challenging exercises and practice questions.",
        "sad": "Consider a short break and review the topic with simpler examples.",
        "angry": "Pause briefly and divide the task into smaller steps.",
        "fear": "Review fundamental concepts using guided examples.",
        "disgust": "Try a different learning activity or explanation.",
        "neutral": "Try an interactive quiz or practical example.",
        "surprise": "Explore the concept further with examples."
    }
    return suggestions.get(emotion.lower(), "Continue learning at your preferred pace.")

st.title("🧠 NeuroVision AI")
st.caption("REAL-TIME FACIAL EXPRESSION RECOGNITION AND ADAPTIVE LEARNING SUPPORT")

with st.sidebar:
    st.subheader("NeuroVision AI")
    page = st.radio("MAIN NAVIGATION", [
        "Dashboard", "Live Emotion Detection", "Emotion Analytics",
        "Learning Support", "Database History"
    ])
    st.divider()
    st.caption("Dataset: FER-2013")
    st.caption("Model: MobileNetV3-Small")
    st.caption("Detection: OpenCV")
    st.caption("Database: SQLite")

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if page == "Dashboard":
    st.header("Project Overview")
    summary = get_database_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Emotion Categories", "07")
    c2.metric("Learning Sessions", summary["total_sessions"])
    c3.metric("Saved Observations", summary["total_observations"])
    st.subheader("Supported Facial Expressions")
    names = [("😠","Angry"),("🤢","Disgust"),("😨","Fear"),("😊","Happy"),
             ("😐","Neutral"),("😢","Sad"),("😲","Surprise")]
    cols = st.columns(4)
    for i, (emoji, name) in enumerate(names):
        with cols[i % 4]:
            st.markdown(f"### {emoji} {name}")
    st.info("The system estimates visible facial expressions; it does not establish a person's actual internal emotional state.")

elif page == "Live Emotion Detection":
    st.header("Live Emotion Intelligence")
    if st.session_state.session_id is None:
        st.info("Start a session before activating the webcam.")
        if st.button("▶ Start Learning Session"):
            st.session_state.session_id = create_session()
            st.rerun()
    else:
        session_id = st.session_state.session_id
        st.success(f"Active Learning Session | ID: {session_id}")
        if st.button("⏹ End Learning Session"):
            end_session(session_id)
            st.session_state.session_id = None
            st.rerun()

        save_state = {"last_saved": 0.0}
        save_lock = threading.Lock()

        def video_frame_callback(frame):
            image = frame.to_ndarray(format="bgr24")
            faces = detector.detect_faces(image)
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                crop = image[max(0,y):y+h, max(0,x):x+w]
                if crop.size:
                    emotion, confidence = predictor.predict(crop)
                    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(image, f"{emotion.upper()} {confidence*100:.1f}%",
                                (x, max(25,y-10)), cv2.FONT_HERSHEY_SIMPLEX,
                                .7, (0,255,0), 2)
                    now = time.monotonic()
                    should_save = False
                    with save_lock:
                        if now - save_state["last_saved"] >= 5:
                            save_state["last_saved"] = now
                            should_save = True
                    if should_save:
                        add_observation(session_id, emotion, confidence,
                                        recommendation(emotion))
            else:
                cv2.putText(image, "NO FACE DETECTED", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,255), 2)
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        webrtc_streamer(
            key=f"neurovision-{session_id}",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        st.caption("An observation is saved approximately every five seconds when a face is detected.")

elif page == "Emotion Analytics":
    st.header("Emotion Analytics")
    records = get_observations()
    if not records:
        st.info("No observations yet. Start a webcam session first.")
    else:
        df = pd.DataFrame(records)
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Observations", len(df))
        c2.metric("Expression Categories", df["emotion"].nunique())
        c3.metric("Average Confidence", f"{df['confidence'].mean()*100:.1f}%")
        counts = df["emotion"].value_counts().rename_axis("Emotion").reset_index(name="Count")
        left,right = st.columns(2)
        with left:
            fig = px.pie(counts, names="Emotion", values="Count", hole=.5,
                         title="Expression Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(counts, x="Emotion", y="Count", color="Emotion",
                          title="Expression Frequency")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download Analytics CSV",
                           df.to_csv(index=False).encode("utf-8"),
                           "emotion_analytics.csv", "text/csv")

elif page == "Learning Support":
    st.header("Adaptive Learning Support")
    records = get_observations()
    if not records:
        st.info("No observations available. Start a learning session first.")
    else:
        latest = records[0]
        st.metric("Latest Expression", latest["emotion"].title())
        st.metric("Prediction Confidence", f'{latest["confidence"]*100:.1f}%')
        st.subheader("Recommended Learning Activity")
        st.success(latest["recommendation"])
        st.caption("Recommendations are optional and based on facial-expression estimates.")

elif page == "Database History":
    st.header("Database Management")
    summary = get_database_summary()
    c1,c2 = st.columns(2)
    c1.metric("Total Learning Sessions", summary["total_sessions"])
    c2.metric("Saved Observations", summary["total_observations"])
    st.subheader("Learning Sessions")
    sessions = get_sessions()
    if sessions:
        st.dataframe(pd.DataFrame(sessions), use_container_width=True, hide_index=True)
    else:
        st.info("No sessions saved yet.")
    st.subheader("Facial Expression Records")
    records = get_observations()
    if records:
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Export Database Records",
                           df.to_csv(index=False).encode("utf-8"),
                           "database_emotion_records.csv", "text/csv")
    else:
        st.info("No observations saved yet.")

st.divider()
st.caption("NeuroVision AI | FER-2013 | MobileNetV3-Small | OpenCV | Streamlit | SQLite")
