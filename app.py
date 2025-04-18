import streamlit as st
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import os
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import io
import subprocess
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path="best.pt"):
    """Load the YOLO model with error handling"""
    try:
        model = YOLO(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Model loading failed: {str(e)}")
        return None

def detect_landslide(model, image, conf_threshold=0.25):
    """Detect landslides with configurable confidence threshold"""
    try:
        results = model(image, conf=conf_threshold, verbose=False)
        detections = []
        for result in results:
            img = result.plot()  # This returns RGB
            for box in result.boxes:
                detections.append({
                    'confidence': float(box.conf),
                    'class': result.names[int(box.cls)],
                    'bbox': box.xyxy[0].tolist()
                })
        return img, detections
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        return image, []  # Return original image if detection fails

def convert_video(input_path, output_path):
    """Convert video to H.264/MP4 using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path, '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'aac', output_path, '-y'
        ]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Video converted successfully: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg conversion failed: {str(e.stderr.decode())}")
        logger.error(f"FFmpeg conversion failed: {str(e.stderr.decode())}")
        return None
    except FileNotFoundError:
        st.error("FFmpeg not found. Ensure FFmpeg is installed (should be handled via packages.txt).")
        logger.error("FFmpeg executable not found in system PATH")
        return None

def process_video(model, video_path, progress_bar, status_text, conf_threshold=0.10):
    """Process video with live display at original resolution, using XVID and converting to MP4"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        status_text.text("Error opening video file")
        logger.error(f"Failed to open video file: {video_path}")
        return None, []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_video_path = f"temp_output_{timestamp}.avi"  # Initial output with XVID
    final_video_path = f"output_detected_{timestamp}.mp4"  # Final MP4 output
    
    # Use XVID codec initially (more reliable for OpenCV writing)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        status_text.text("Error initializing video writer with XVID")
        logger.error("VideoWriter failed to initialize with XVID")
        cap.release()
        return None, []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        status_text.text("Error: Video has no frames")
        logger.error("Video has no frames to process")
        cap.release()
        return None, []
    
    processed_frames = 0
    all_detections = []
    
    status_text.text(f"Processing video live... {frame_width}x{frame_height} at {fps} FPS")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_frame, detections = detect_landslide(model, frame_rgb, conf_threshold)
        
        result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
        out.write(result_frame_bgr)
        
        processed_frames += 1
        all_detections.extend(detections)
        
        progress_percentage = int((processed_frames / total_frames) * 100)
        progress_bar.progress(progress_percentage / 100)
        status_text.text(f"Processing Video: {progress_percentage}%")
    
    cap.release()
    out.release()
    time.sleep(1)  # Ensure file is written
    
    if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
        status_text.text("Error: Initial video writing failed - output file is empty or not created")
        logger.error(f"Initial video file is empty or not created: {temp_video_path}")
        return None, all_detections
    
    # Convert the video to MP4 with H.264
    status_text.text("Converting video to MP4...")
    converted_path = convert_video(temp_video_path, final_video_path)
    
    if converted_path and os.path.exists(converted_path) and os.path.getsize(converted_path) > 0:
        status_text.text(f"Video processing and conversion completed: {converted_path}")
        # Clean up the temporary AVI file
        try:
            os.unlink(temp_video_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {temp_video_path}: {str(e)}")
        return converted_path, all_detections
    else:
        status_text.text("Error: Video conversion failed, falling back to AVI")
        logger.warning(f"Video conversion failed, using fallback: {temp_video_path}")
        return temp_video_path, all_detections

def plot_confidence_distribution(detections):
    """Plot confidence score distribution"""
    if not detections:
        return None
    confidences = [d['confidence'] for d in detections]
    fig, ax = plt.subplots()
    ax.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Confidence Score Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def calculate_pr_curves(detections, num_thresholds=100):
    """Calculate precision, recall, and F1 at different confidence thresholds"""
    if not detections:
        return None, None, None, None
    
    thresholds = np.linspace(0, 1, num_thresholds)
    precisions = []
    recalls = []
    f1_scores = []
    
    total_positives = len(detections)
    
    for threshold in thresholds:
        true_positives = sum(1 for d in detections if d['confidence'] >= threshold)
        false_positives = 0  # Simplified assumption
        false_negatives = total_positives - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    return thresholds, precisions, recalls, f1_scores

def plot_pr_curve(thresholds, precisions, recalls, current_threshold=None):
    """Plot precision-recall curve with current threshold marker"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, color='blue', marker='.', label='Precision-Recall curve')
    
    if current_threshold is not None:
        threshold_idx = (np.abs(thresholds - current_threshold)).argmin()
        current_precision = precisions[threshold_idx]
        current_recall = recalls[threshold_idx]
        ax.scatter([current_recall], [current_precision], color='red', s=100, zorder=5, 
                  label=f'Current threshold ({current_threshold:.2f})')
    
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True)
    ax.legend()
    
    pr_auc = auc(recalls, precisions) if len(recalls) > 1 else 0
    ax.text(0.5, 0.2, f'AUC = {pr_auc:.3f}', fontsize=12)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def plot_p_curve(thresholds, precisions, current_threshold=None):
    """Plot precision vs threshold curve with current threshold marker"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, precisions, color='green', marker='.', label='Precision curve')
    
    if current_threshold is not None:
        threshold_idx = (np.abs(thresholds - current_threshold)).argmin()
        current_precision = precisions[threshold_idx]
        ax.scatter([current_threshold], [current_precision], color='red', s=100, zorder=5, 
                  label=f'Current threshold ({current_threshold:.2f})')
    
    ax.set_title('Precision vs. Confidence Threshold')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Precision')
    ax.grid(True)
    ax.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def plot_r_curve(thresholds, recalls, current_threshold=None):
    """Plot recall vs threshold curve with current threshold marker"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, recalls, color='red', marker='.', label='Recall curve')
    
    if current_threshold is not None:
        threshold_idx = (np.abs(thresholds - current_threshold)).argmin()
        current_recall = recalls[threshold_idx]
        ax.scatter([current_threshold], [current_recall], color='blue', s=100, zorder=5, 
                  label=f'Current threshold ({current_threshold:.2f})')
    
    ax.set_title('Recall vs. Confidence Threshold')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Recall')
    ax.grid(True)
    ax.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def show_curve_explanations():
    """Display explanations for P, PR, and R curves"""
    with st.expander("🔍 Understanding the Performance Curves", expanded=False):
        st.markdown("""
        ### Precision-Recall (PR) Curve
        
        **What it shows:** The trade-off between precision and recall at different confidence thresholds.
        
        **How to interpret it:**
        - A curve that stays high (toward the top-right corner) indicates a better model
        - Area Under the Curve (AUC) is a single metric summarizing performance - higher is better
        - A perfect detector would have an AUC of 1.0
        
        **Practical use:** This curve helps you understand how changing the confidence threshold affects the balance between false positives and false negatives in landslide detection.
        
        ---
        
        ### Precision Curve (P-Curve)
        
        **What it shows:** How precision changes as the confidence threshold increases.
        
        **How to interpret it:**
        - Precision typically increases with higher thresholds (model becomes more selective)
        - A sharp increase indicates a potentially good threshold cutoff
        
        **Practical use:** Set thresholds to minimize false landslide detections when false alarms would be costly.
        
        ---
        
        ### Recall Curve (R-Curve)
        
        **What it shows:** How recall changes as the confidence threshold increases.
        
        **How to interpret it:**
        - Recall typically decreases with higher thresholds (model detects fewer examples)
        - A curve that stays high even at higher thresholds indicates a robust detector
        
        **Practical use:** Set thresholds to ensure you catch most landslides when missing a detection would be dangerous.
        
        ---
        
        ### How to Use These Curves Together
        
        1. **High-risk areas:** For areas where missing a landslide is dangerous, prioritize recall (use lower thresholds)
        2. **Resource-limited scenarios:** When response resources are limited, prioritize precision (use higher thresholds)
        3. **Balanced approach:** Look for the threshold that maximizes F1 score (harmonic mean of precision and recall)
        
        The current threshold is marked with a point on each curve to help you see how your selected threshold performs.
        """)

def filter_detections(all_detections, threshold):
    """Filter detections based on the confidence threshold"""
    return [d for d in all_detections if d['confidence'] >= threshold]

def show_detection_results(detection_data, current_threshold=0.25, export_csv=False, download_enabled=False, debug_mode=False, image_shape=None, frame_dims=None):
    """Display detection statistics and results with interactive threshold control"""
    all_detections = detection_data.copy() if detection_data else []
    
    if all_detections:
        st.markdown("---")
        st.subheader("🎚️ Confidence Threshold Level")
        
        threshold = st.slider(
            " ",
            min_value=0.05,
            max_value=0.95,
            value=current_threshold,
            step=0.05,
        )
        
        filtered_detections = filter_detections(all_detections, threshold)
        
        st.markdown("---")
        st.subheader("📈 Detection Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections (All)", len(all_detections))
        with col2:
            st.metric("Filtered Detections", len(filtered_detections))
        with col3:
            reduction = ((len(all_detections) - len(filtered_detections)) / len(all_detections) * 100) if len(all_detections) > 0 else 0
            st.metric("Reduction %", f"{reduction:.1f}%")
        
        if filtered_detections:
            df = pd.DataFrame(filtered_detections)
            st.dataframe(df)
        else:
            st.warning("No detections above the current threshold.")
        
        conf_plot = plot_confidence_distribution(all_detections)
        if conf_plot:
            st.image(conf_plot, caption="Confidence Score Distribution (All Detections)", use_container_width=True)
        
        if export_csv and not df.empty and download_enabled:
            csv_buffer = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Detections CSV", 
                             csv_buffer, 
                             file_name=f"detections_threshold_{threshold:.2f}.csv", 
                             mime="text/csv")
    
        st.markdown("---")
        st.subheader("📋 Detection Summary")
        
        if filtered_detections:
            avg_conf = np.mean([d['confidence'] for d in filtered_detections])
            st.write(f"Average Confidence of Filtered Detections: {avg_conf:.2%}")
            
            if 'class' in filtered_detections[0]:
                class_counts = {}
                for d in filtered_detections:
                    class_counts[d['class']] = class_counts.get(d['class'], 0) + 1
                
                st.write("Class Distribution:")
                for cls, count in class_counts.items():
                    st.write(f"- {cls}: {count} ({count/len(filtered_detections):.1%})")
        
        show_curve_explanations()
        
        st.markdown("---")
        st.subheader("🔍 Precision-Recall Analysis")
        
        thresholds, precisions, recalls, f1_scores = calculate_pr_curves(all_detections)
        
        if thresholds is not None:
            col1, col2 = st.columns(2)
            
            pr_plot = plot_pr_curve(thresholds, precisions, recalls, threshold)
            col1.image(pr_plot, caption="Precision-Recall Curve", use_container_width=True)
            
            p_plot = plot_p_curve(thresholds, precisions, threshold)
            col2.image(p_plot, caption="Precision Curve", use_container_width=True)
            
            r_plot = plot_r_curve(thresholds, recalls, threshold)
            st.image(r_plot, caption="Recall Curve", use_container_width=True)
            
            best_f1_idx = np.argmax(f1_scores)
            best_f1_threshold = thresholds[best_f1_idx]
            best_f1 = f1_scores[best_f1_idx]
            
            st.markdown("---")
            st.subheader("💡 Threshold Recommendations")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best F1 Score Threshold", f"{best_f1_threshold:.2f}")
                if st.button("Apply Best F1"):
                    st.experimental_set_query_params(threshold=best_f1_threshold)
                    st.experimental_rerun()
            
            with col2:
                high_prec_indices = [i for i, p in enumerate(precisions) if p > 0.9]
                high_prec_threshold = thresholds[high_prec_indices[0]] if high_prec_indices else 0.7
                st.metric("High Precision Threshold", f"{high_prec_threshold:.2f}")
                if st.button("Apply High Precision"):
                    st.experimental_set_query_params(threshold=high_prec_threshold)
                    st.experimental_rerun()
            
            with col3:
                high_recall_indices = [i for i, r in enumerate(recalls) if r > 0.9]
                high_recall_threshold = thresholds[high_recall_indices[-1]] if high_recall_indices else 0.2
                st.metric("High Recall Threshold", f"{high_recall_threshold:.2f}")
                if st.button("Apply High Recall"):
                    st.experimental_set_query_params(threshold=high_recall_threshold)
                    st.experimental_rerun()
    
        if debug_mode:
            st.markdown("---")
            st.subheader("🛠️ Debug Information")
            st.write(f"Model Path: best.pt")
            if image_shape:
                st.write(f"Input Resolution: {image_shape}")
            elif frame_dims:
                st.write(f"Input Resolution: {frame_dims}")
            st.write(f"Current Confidence Threshold: {threshold}")
            st.write(f"All Detections Found: {len(all_detections)}")
            st.write(f"Filtered Detections: {len(filtered_detections)}")
            if not all_detections:
                st.warning("No detections found. Check input quality or model training.")

def main():
    st.set_page_config(layout="wide", page_title="Landslide Detection App")
    st.markdown("## 🌍 Landslide Detection App")
    
    query_params = st.query_params
    default_threshold = float(query_params.get("threshold", [0.10])[0])
    
    model = load_model()
    if model is None:
        return
    
    with st.sidebar:
        st.header(" ☰ Home Menu")
        
        theme = st.selectbox("🎨 Theme", ["Light", "Dark"])
        if theme == "Dark":
            st.markdown("""
                <style>
                .stApp {
                    background-color: #1E1E1E;
                    color: white;
                }
                </style>
                """, unsafe_allow_html=True)
        
        st.subheader("📥 Input Selection")
        input_type = st.radio("Select Input Type:", ("Image", "Video"))
        
        conf_threshold = st.slider(
            "Detection Confidence",
            min_value=0.05,
            max_value=0.95,
            value=default_threshold,
            step=0.05,
            help="Initial confidence threshold for detection (can be adjusted later)"
        )
        
        if input_type == "Image":
            uploaded_file = st.file_uploader("📤 Upload Image", 
                                          type=["jpg", "png", "jpeg"], 
                                          accept_multiple_files=False)
        else:
            uploaded_file = st.file_uploader("📤 Upload Video", 
                                          type=["mp4", "avi", "mov"], 
                                          accept_multiple_files=False)
        
        if uploaded_file is not None:
            file_size = uploaded_file.size / (1024 * 1024)
            st.write(f"File Size: {file_size:.2f} MB")
        
        st.markdown("---")
        debug_mode = st.checkbox("Debug Mode", value=False, 
                               help="Show detailed detection info and logs")
        
        st.markdown("---")
        st.subheader("💾 Download Options")
        download_enabled = st.checkbox("Enable Download", value=True)
        export_csv = st.checkbox("Export Detections as CSV", value=False)
        
        st.markdown("---")
        st.subheader("📊 Information")
        st.write("- Includes P-curve, PR-curve, and R-curve visualizations.")
        st.write("- Adjust thresholds to optimize performance.")
    
    if uploaded_file is None:
        st.info("👈 Please upload a file in the sidebar to begin detection.")
        
        with st.expander("❓ Help & Instructions"):
            st.write("""
            - Select input type (Image or Video) in the sidebar.
            - Set your initial confidence threshold.
            - Upload your file using the uploader.
            - The system will automatically process your input.
            - You can adjust the threshold after processing to see how it affects detections.
            - Use the performance curves to select optimal threshold values.
            """)
            
        show_curve_explanations()
    else:
        if input_type == "Image":
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🖼️ Original Image")
                st.image(image_np, use_container_width=True)
            
            process_button = st.button("Process Image", key="process_image")
            
            if process_button:
                with st.spinner("🔍 Detecting..."):
                    start_time = time.time()
                    result_image, detections = detect_landslide(model, image_np, conf_threshold)
                    processing_time = time.time() - start_time
                    st.sidebar.write(f"Processing Time: {processing_time:.2f} seconds")
                
                with col2:
                    st.subheader("✅ Detected Image")
                    st.image(result_image, use_container_width=True)
                    
                    if download_enabled:
                        img_buffer = cv2.imencode('.png', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))[1].tobytes()
                        st.download_button("Download Detected Image", 
                                         img_buffer, 
                                         file_name="detected_image.png", 
                                         mime="image/png")
                
                show_detection_results(detections, conf_threshold, export_csv, download_enabled, debug_mode, image_np.shape)
        
        else:  # Video processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_video_path = temp_file.name
            
            frame_dims = None
            cap = cv2.VideoCapture(temp_video_path)
            if cap.isOpened():
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_dims = (frame_width, frame_height)
                cap.release()
            
            st.subheader("🎥 Original Video")
            st.video(temp_video_path)
            
            process_button = st.button("Process Video", key="process_video")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if process_button:
                start_time = time.time()
                processed_video_path, detections = process_video(model, temp_video_path, progress_bar, 
                                                             status_text, conf_threshold)
                processing_time = time.time() - start_time
                
                if processed_video_path and os.path.exists(processed_video_path):
                    st.sidebar.write(f"Processing Time: {processing_time:.2f} seconds")
                    st.sidebar.write(f"Output Size: {os.path.getsize(processed_video_path)/(1024*1024):.2f} MB")
                    
                    st.subheader("📹 Detected Video")
                    try:
                        with open(processed_video_path, 'rb') as video_file:
                            st.video(video_file, format="video/mp4" if processed_video_path.endswith('.mp4') else "video/avi")
                    except Exception as e:
                        st.error(f"Error displaying video: {str(e)}")
                        logger.error(f"Video display error: {str(e)}")
                        st.info("Try downloading the video and playing it locally to verify its content.")
                    
                    if download_enabled:
                        with open(processed_video_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                        st.download_button("Download Detected Video", 
                                         video_bytes, 
                                         file_name=os.path.basename(processed_video_path),
                                         mime="video/mp4" if processed_video_path.endswith('.mp4') else "video/avi",
                                         key="download_video")
                    
                    show_detection_results(detections, conf_threshold, export_csv, download_enabled, debug_mode, frame_dims=frame_dims)
                    
                    try:
                        os.unlink(temp_video_path)
                        if os.path.exists(processed_video_path):
                            os.unlink(processed_video_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp files: {str(e)}")
                else:
                    st.error("Video processing failed. Check debug mode for details.")
    
    st.markdown("---")
    st.markdown("""
        <footer style='text-align: center; color: gray;'>
            Landslide Detection App v1.6
        </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
