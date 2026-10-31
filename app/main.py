
import pandas as pd
import time
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
    page_icon="OD",
    layout="wide"
)


# ==========================================
# 2. LOAD YOLO MODEL
# ==========================================

@st.cache_resource
def load_model(model_name):

    model_path = (
        Path(__file__).resolve().parent.parent
        / f"{model_name}.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

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

    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

    def __init__(self, model_name):

        self.model = load_model(model_name)

        # Performance monitoring
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.start_time = time.perf_counter()
        # Store movement history for each tracking ID
        self.track_history = {}

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        # Measure inference time
        inference_start = time.perf_counter()

        results = self.model.track(
            image,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False
        )

        inference_end = time.perf_counter()

        inference_time = inference_end - inference_start
        self.total_inference_time += inference_time
        self.frame_count += 1

        # Calculate performance metrics
        average_inference_time = (
            self.total_inference_time / self.frame_count
        ) * 1000

        elapsed_time = time.perf_counter() - self.start_time

        actual_fps = (
            self.frame_count / elapsed_time
            if elapsed_time > 0
            else 0
        )

        result = results[0]
        annotated_image = result.plot()

        # Draw movement trails for tracked objects
        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):

                x1, y1, x2, y2 = map(int, box)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if track_id not in self.track_history:
                    self.track_history[track_id] = []

                self.track_history[track_id].append(
                    (center_x, center_y)
                )

                # Keep only the latest 30 positions
                self.track_history[track_id] = (
                    self.track_history[track_id][-30:]
                )

                points = self.track_history[track_id]

                # Draw movement trail
                if len(points) > 1:

                    for i in range(1, len(points)):

                        cv2.line(
                            annotated_image,
                            points[i - 1],
                            points[i],
                            (255, 0, 255),
                            2
                        )

        count = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        # Display object count on every frame
        cv2.putText(
            annotated_image,
            f"Objects detected: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Display performance metrics
        cv2.putText(
            annotated_image,
            f"Inference: {average_inference_time:.2f} ms",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_image,
            f"FPS: {actual_fps:.2f}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_image,
            f"Frames: {self.frame_count}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            annotated_image,
            format="bgr24"
        )



# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Navigation")

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

st.sidebar.divider()

st.sidebar.subheader("Detection Model")

selected_model = st.sidebar.selectbox(
    "Detection model",
    ["yolo11n", "yolo11s"],
    format_func=lambda x: (
        "YOLO11n (faster)"
        if x == "yolo11n"
        else "YOLO11s (larger model)"
    )
)

st.sidebar.caption(
    "YOLO11n is the lighter model. YOLO11s is larger and may detect some objects more reliably, but can run slower."
)


# ==========================================
# 6. APPLICATION HEADER
# ==========================================

st.title("Intelligent Object Detection System")

st.caption(
    "Object detection and tracking with YOLO and OpenCV"
)

st.divider()


# ==========================================
# 7. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.header("Project Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Detection Model",
        selected_model.upper()
    )

    col2.metric(
        "Image Processing",
        "OpenCV"
    )

    col3.metric(
        "Interface",
        "Streamlit"
    )

    st.subheader("Welcome")

    st.write(
        "Use this application to detect objects in images, process recorded videos, "
        "and test detection through a live webcam."
    )

    st.info(
        "Choose a section from the sidebar to begin."
    )

    st.subheader("Available Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown(
            """
            **Image Detection**

            - Upload images
            - Detect multiple objects
            - View bounding boxes and confidence
            - Download detection results

            **Video Detection**

            - Upload videos
            - Detect and track objects
            - Track unique object IDs
            - Download processed videos
            """
        )

    with feature_col2:
        st.markdown(
            """
            **Live Webcam**

            - Real-time object detection
            - Object tracking
            - Live bounding boxes

            **Analytics**

            - Detection history
            - Session statistics
            - Object detection charts
            """
        )


# ==========================================
# 8. IMAGE DETECTION
# ==========================================

elif page == "Image Detection":

    st.header("Image Detection")

    st.write(
        "Choose an image to view the detected objects and their counts."
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
                f"Loading {selected_model.upper()} and detecting objects..."
            ):

                model = load_model(selected_model)

                # Model warm-up (excluded from performance measurement)
                model.predict(
                    image,
                    conf=confidence,
                    verbose=False
                )

                # Actual inference measurement
                start_time = time.perf_counter()

                results = model.predict(
                    image,
                    conf=confidence,
                    verbose=False
                )

                end_time = time.perf_counter()

                inference_time = (end_time - start_time) * 1000

                inference_fps = (
                    1000 / inference_time
                    if inference_time > 0
                    else 0
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

            # Save image detection result once
            # for the same image, model and confidence.

            image_id = hashlib.md5(
                uploaded_file.getvalue()
            ).hexdigest()

            record_id = (
                f"{image_id}_{selected_model}_{confidence}"
            )

            if (
                st.session_state.get("last_image_recorded")
                != record_id
            ):

                save_detection_record({
                    "type": "image",
                    "model": selected_model,
                    "total_objects": total_objects,
                    "object_counts": object_counts
                })

                st.session_state["last_image_recorded"] = record_id

            # Object Analysis

            st.subheader("Object Analysis")

            col1, col2, col3 = st.columns(3)

            st.subheader("Performance")

            perf_col1, perf_col2 = st.columns(2)

            perf_col1.metric(
                "Inference Time",
                f"{inference_time:.2f} ms"
            )

            perf_col2.metric(
                "Estimated Processing Speed",
                f"{inference_fps:.2f} FPS"
            )

            col1.metric(
                "Total Objects Detected",
                total_objects
            )

            col2.metric(
                "Different Object Classes",
                len(object_counts)
            )

            col3.metric(
                "Selected Model",
                selected_model.upper()
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

    st.header("Video Detection and Tracking")

    st.write(
        "Choose a video to detect objects and follow their movement across frames."
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

        current_video_id = hashlib.md5(
            uploaded_video.getvalue()
        ).hexdigest()

        if (
            st.session_state.get("processed_video_source")
            != current_video_id
        ):
            st.session_state.pop("processed_video", None)
            st.session_state.pop("processed_heatmap", None)
            st.session_state["processed_video_source"] = current_video_id

        st.video(uploaded_video)

        if st.button("Start Video Detection"):

            with st.spinner(
                f"Loading {selected_model.upper()}..."
            ):

                model = load_model(selected_model)

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

                track_history = {}
                heatmap_accumulator = np.zeros((height, width), dtype=np.float32)
                last_frame = None

                total_inference_time = 0.0

                video_start_time = time.perf_counter()

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    inference_start = time.perf_counter()

                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=confidence,
                        verbose=False
                    )

                    inference_end = time.perf_counter()

                    total_inference_time += (
                        inference_end - inference_start
                    )

                    result = results[0]
                    last_frame = frame.copy()

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

                    # Draw movement trails for tracked objects
                    if result.boxes is not None and result.boxes.id is not None:

                        boxes = result.boxes.xyxy.cpu().numpy()
                        track_ids = result.boxes.id.int().cpu().tolist()

                        for box, track_id in zip(boxes, track_ids):

                            x1, y1, x2, y2 = map(int, box)

                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2

                            # Add this tracked position to the heatmap.
                            cv2.circle(
                                heatmap_accumulator,
                                (center_x, center_y),
                                20,
                                1.0,
                                -1
                            )

                            if track_id not in track_history:
                                track_history[track_id] = []

                            track_history[track_id].append(
                                (center_x, center_y)
                            )

                            # Keep only the last 30 positions
                            track_history[track_id] = (
                                track_history[track_id][-30:]
                            )

                            points = track_history[track_id]

                            # Draw the movement trail
                            if len(points) > 1:

                                for i in range(1, len(points)):

                                    cv2.line(
                                        annotated_frame,
                                        points[i - 1],
                                        points[i],
                                        (255, 0, 255),
                                        2
                                    )

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

                # Finalize the output file before it is read or displayed.
                cap.release()
                writer.release()

                video_end_time = time.perf_counter()

                total_processing_time = (
                    video_end_time - video_start_time
                )

                average_inference_time = (
                    (total_inference_time / processed_frames) * 1000
                    if processed_frames > 0
                    else 0
                )

                actual_fps = (
                    processed_frames / total_processing_time
                    if total_processing_time > 0
                    else 0
                )

                # Save video detection history

                if processed_frames > 0:

                    save_detection_record({
                        "type": "video",
                        "model": selected_model,
                        "total_objects": sum(object_counts.values()),
                        "object_counts": object_counts,
                        "frames_processed": processed_frames,
                        "unique_tracking_ids": len(unique_ids)
                    })

                    st.success(
                        "Video detection completed!"
                    )

                    st.subheader("Video Analysis")

                    col1, col2, col3, col4 = st.columns(4)
                    st.subheader("Video Performance")

                    perf_col1, perf_col2, perf_col3 = st.columns(3)

                    perf_col1.metric(
                        "Average Inference Time",
                        f"{average_inference_time:.2f} ms/frame"
                    )

                    perf_col2.metric(
                        "Actual Processing FPS",
                        f"{actual_fps:.2f}"
                    )

                    perf_col3.metric(
                        "Total Processing Time",
                        f"{total_processing_time:.2f} seconds"
                    )

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

                    col4.metric(
                        "Selected Model",
                        selected_model.upper()
                    )

                    st.subheader(
                        "Class-wise Detection Count"
                    )

                    for name, count in object_counts.items():
                        st.write(f"- {name}: {count}")

                    with open(output_path, "rb") as video_file:
                        st.session_state["processed_video"] = video_file.read()

                    # Convert accumulated tracked positions into a heatmap.
                    if last_frame is not None and np.any(heatmap_accumulator):
                        normalized_heatmap = cv2.normalize(
                            heatmap_accumulator,
                            None,
                            0,
                            255,
                            cv2.NORM_MINMAX
                        ).astype(np.uint8)

                        colored_heatmap = cv2.applyColorMap(
                            normalized_heatmap,
                            cv2.COLORMAP_JET
                        )

                        heatmap_overlay = cv2.addWeighted(
                            last_frame,
                            0.6,
                            colored_heatmap,
                            0.4,
                            0
                        )

                        heatmap_success, encoded_heatmap = cv2.imencode(
                            ".png",
                            heatmap_overlay
                        )

                        if heatmap_success:
                            st.session_state["processed_heatmap"] = encoded_heatmap.tobytes()
                    else:
                        st.session_state.pop("processed_heatmap", None)

                else:

                    st.error(
                        "No frames could be processed."
                    )

                status.empty()

        # Keep the processed video visible after Streamlit reruns,
        # including the rerun caused by clicking the download button.
        if st.session_state.get("processed_video"):

            st.subheader("Processed Video")

            st.video(st.session_state["processed_video"])

            st.download_button(
                "Download Processed Video",
                data=st.session_state["processed_video"],
                file_name="tracked_output.mp4",
                mime="video/mp4",
                key="download_processed_video"
            )

        if st.session_state.get("processed_heatmap"):
            st.subheader("Object Movement Heatmap")
            st.caption(
                "Warmer colours indicate areas where tracked object centres "
                "were recorded more frequently during the video."
            )
            st.image(
                st.session_state["processed_heatmap"],
                caption="Movement heatmap overlay on the final video frame",
                use_container_width=True
            )
            st.download_button(
                "Download Movement Heatmap",
                data=st.session_state["processed_heatmap"],
                file_name="movement_heatmap.png",
                mime="image/png",
                key="download_movement_heatmap"
            )


# ==========================================
# 10. LIVE WEBCAM DETECTION
# ==========================================

elif page == "Live Webcam":

    st.header("Live Webcam Detection")

    st.write(
        "Use your laptop camera to detect and track objects in real time."
    )

    st.info(
        "If prompted, allow your browser to access the camera."
    )

    webrtc_streamer(
    key=f"live-object-detection-{selected_model}",
    video_processor_factory=lambda: YOLOVideoProcessor(
        selected_model
    ),
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    async_processing=True
)


# ==========================================
# 11. ANALYTICS
# ==========================================

elif page == "Analytics":

    st.header("Detection Analytics")

    history = load_detection_history()

    if not history:

        st.info(
            "No detection history available. "
            "Please perform image or video detection first."
        )

    else:

        df = pd.DataFrame(history)

        total_sessions = len(df)

        total_detections = int(
            df["total_objects"].sum()
        )

        image_sessions = len(
            df[df["type"] == "image"]
        )

        video_sessions = len(
            df[df["type"] == "video"]
        )

        # Summary metrics

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Sessions",
            total_sessions
        )

        col2.metric(
            "Total Objects",
            total_detections
        )

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

        display_df = df.copy()

        if "object_counts" in display_df.columns:

            display_df["object_counts"] = display_df[
                "object_counts"
            ].apply(
                lambda x: (
                    json.dumps(x)
                    if isinstance(x, dict)
                    else x
                )
            )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Charts side by side

        chart1, chart2 = st.columns(2)

        with chart1:

            st.subheader("📊 Sessions by Type")

            session_counts = (
                df["type"]
                .value_counts()
                .rename_axis("Detection Type")
                .rename("Sessions")
            )

            st.bar_chart(
                session_counts,
                height=280
            )

        with chart2:

            st.subheader("🎯 Objects per Session")

            session_objects = df.set_index(
                "timestamp"
            )["total_objects"]

            st.bar_chart(
                session_objects,
                height=280
            )

        # Export analytics

        st.divider()

        st.subheader("Export Detection History")

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download History as CSV",
            data=csv_data,
            file_name="detection_history.csv",
            mime="text/csv"
        )
