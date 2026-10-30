
import pandas as pd
import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path

import av
import streamlit as st
import cv2
import numpy as np

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Intelligent Object Detection",
    page_icon="🎯",
    layout="wide"
)


# ==========================================
# 2. LOAD YOLO MODEL
# ==========================================

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent.parent / "yolo11n.pt"
    return YOLO(str(model_path))


# ==========================================
# 3. DETECTION HISTORY MANAGEMENT
# ==========================================

HISTORY_FILE = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "detection_history.json"
)


def save_detection_record(record):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    history = load_detection_history()

    record["timestamp"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    history.append(record)

    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)


def load_detection_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return []


# ==========================================
# 4. LIVE WEBCAM PROCESSOR
# ==========================================

class YOLOVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = load_model()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        results = self.model.track(
            image,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False
        )

        result = results[0]
        annotated_image = result.plot()

        count = len(result.boxes) if result.boxes is not None else 0

        cv2.putText(
            annotated_image,
            f"Objects detected: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(
            annotated_image,
            format="bgr24"
        )


# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🎯 Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Image Detection",
        "Video Detection",
        "Live Webcam",
        "Analytics"
    ]
)


# ==========================================
# 6. APPLICATION HEADER
# ==========================================

st.title("🎯 Intelligent Object Detection System")

st.caption(
    "Real-Time Object Detection using Deep Learning"
)

st.divider()


# ==========================================
# 7. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.header("Project Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Detection Model", "YOLO11n")
    col2.metric("Framework", "OpenCV")
    col3.metric("Interface", "Streamlit")

    st.subheader("Welcome")

    st.write(
        "This application performs object detection "
        "and tracking using images, videos and live webcam streams."
    )

    st.info(
        "Select a module from the sidebar to get started."
    )


# ==========================================
# 8. IMAGE DETECTION
# ==========================================

elif page == "Image Detection":

    st.header("📷 Image Detection")

    st.write(
        "Upload an image to detect and count objects."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        key="image_confidence"
    )

    if uploaded_file is not None:

        image_bytes = np.frombuffer(
            uploaded_file.getvalue(),
            np.uint8
        )

        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error("Unable to read this image.")

        else:

            with st.spinner(
                "Loading YOLO model and detecting objects..."
            ):

                model = load_model()

                results = model.predict(
                    image,
                    conf=confidence,
                    verbose=False
                )

            result = results[0]

            annotated_image = result.plot()

            annotated_rgb = cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            )

            original_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            st.subheader("Detection Results")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    original_rgb,
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:
                st.image(
                    annotated_rgb,
                    caption="Detected Objects",
                    use_container_width=True
                )

            # Class-wise object counting

            object_counts = {}

            if result.boxes is not None:

                for cls in result.boxes.cls:

                    class_name = result.names[int(cls)]

                    object_counts[class_name] = (
                        object_counts.get(class_name, 0) + 1
                    )

            total_objects = sum(object_counts.values())

            # Save image detection result only once
            # for the same image and confidence value.

            image_id = hashlib.md5(
                uploaded_file.getvalue()
            ).hexdigest()

            record_id = f"{image_id}_{confidence}"

            if st.session_state.get("last_image_recorded") != record_id:

                save_detection_record({
                    "type": "image",
                    "total_objects": total_objects,
                    "object_counts": object_counts
                })

                st.session_state["last_image_recorded"] = record_id

            # Object Analysis

            st.subheader("Object Analysis")

            col1, col2 = st.columns(2)

            col1.metric(
                "Total Objects Detected",
                total_objects
            )

            col2.metric(
                "Different Object Classes",
                len(object_counts)
            )

            if object_counts:

                st.write("**Class-wise Object Count**")

                for name, count in object_counts.items():
                    st.write(f"- {name}: {count}")

            else:

                st.warning(
                    "No objects detected in this image."
                )

            # Download detection result

            success, encoded_image = cv2.imencode(
                ".jpg",
                annotated_image
            )

            if success:

                st.download_button(
                    "Download Detection Result",
                    data=encoded_image.tobytes(),
                    file_name="detected_image.jpg",
                    mime="image/jpeg"
                )


# ==========================================
# 9. VIDEO DETECTION AND TRACKING
# ==========================================

elif page == "Video Detection":

    st.header("🎥 Video Detection & Tracking")

    st.write(
        "Upload a video to detect and track objects."
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )

    confidence = st.slider(
        "Detection Confidence",
        0.0,
        1.0,
        0.25,
        0.05,
        key="video_confidence"
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("Start Video Detection"):

            model = load_model()

            # Save uploaded video temporarily

            with tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            ) as input_file:

                input_file.write(
                    uploaded_video.getvalue()
                )

                input_path = input_file.name

            output_file = tempfile.NamedTemporaryFile(
                suffix=".mp4",
                delete=False
            )

            output_path = output_file.name
            output_file.close()

            cap = cv2.VideoCapture(input_path)

            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 25

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            if not cap.isOpened() or not writer.isOpened():

                st.error(
                    "Unable to process this video."
                )

                cap.release()
                writer.release()

            else:

                progress = st.progress(0)
                status = st.empty()

                unique_ids = set()
                object_counts = {}
                processed_frames = 0

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=confidence,
                        verbose=False
                    )

                    result = results[0]

                    if result.boxes is not None:

                        for cls in result.boxes.cls:

                            name = result.names[int(cls)]

                            object_counts[name] = (
                                object_counts.get(name, 0) + 1
                            )

                        if result.boxes.id is not None:

                            ids = (
                                result.boxes.id.int().cpu().tolist()
                            )

                            unique_ids.update(ids)

                    annotated_frame = result.plot()

                    writer.write(annotated_frame)

                    processed_frames += 1

                    if total_frames > 0:

                        progress.progress(
                            min(
                                processed_frames / total_frames,
                                1.0
                            )
                        )

                    status.write(
                        f"Processed frames: {processed_frames}"
                    )

                cap.release()
                writer.release()

                # Save video detection history

                if processed_frames > 0:

                    save_detection_record({
                        "type": "video",
                        "total_objects": sum(object_counts.values()),
                        "object_counts": object_counts,
                        "frames_processed": processed_frames,
                        "unique_tracking_ids": len(unique_ids)
                    })

                    st.success(
                        "Video detection completed!"
                    )

                    st.subheader("Video Analysis")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Frames Processed",
                        processed_frames
                    )

                    col2.metric(
                        "Unique Tracking IDs",
                        len(unique_ids)
                    )

                    col3.metric(
                        "Object Detections",
                        sum(object_counts.values())
                    )

                    st.subheader(
                        "Class-wise Detection Count"
                    )

                    for name, count in object_counts.items():
                        st.write(f"- {name}: {count}")

                    st.subheader("Processed Video")

                    st.video(output_path)

                    with open(output_path, "rb") as video_file:

                        st.download_button(
                            "Download Processed Video",
                            data=video_file.read(),
                            file_name="tracked_output.mp4",
                            mime="video/mp4"
                        )

                else:

                    st.error(
                        "No frames could be processed."
                    )

                status.empty()


# ==========================================
# 10. LIVE WEBCAM DETECTION
# ==========================================

elif page == "Live Webcam":

    st.header("📹 Live Webcam Object Detection")

    st.write(
        "Real-time object detection and tracking "
        "using your laptop camera."
    )

    st.info(
        "Allow camera permission when your browser asks for it."
    )

    webrtc_streamer(
        key="live-object-detection",
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )


# ==========================================
# 11. ANALYTICS
# ==========================================

elif page == "Analytics":

    st.header("📊 Detection Analytics")

    history = load_detection_history()

    if not history:

        st.info(
            "No detection history available. "
            "Please perform image or video detection first."
        )

    else:

        df = pd.DataFrame(history)

        total_sessions = len(df)

        total_detections = df["total_objects"].sum()

        image_sessions = len(
            df[df["type"] == "image"]
        )

        video_sessions = len(
            df[df["type"] == "video"]
        )

        # Summary metrics

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Detection Sessions",
            total_sessions
        )

        col2.metric(
            "Total Objects Detected",
            int(total_detections)
        )

        col3, col4 = st.columns(2)

        col3.metric(
            "Image Sessions",
            image_sessions
        )

        col4.metric(
            "Video Sessions",
            video_sessions
        )

        st.divider()

        # Detection history

        st.subheader("📋 Detection History")

        st.dataframe(
            df,
            use_container_width=True
        )

        st.divider()

        # Detection type chart

        st.subheader("📊 Detection Sessions by Type")

        st.bar_chart(
            df["type"].value_counts()
        )

        # Object count chart

        st.subheader("🎯 Objects Detected per Session")

        st.bar_chart(
            df.set_index("timestamp")["total_objects"]
        )
