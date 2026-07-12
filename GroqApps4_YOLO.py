import base64
import io
import json
from groq import Groq
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Crop Disease Analyzer", layout="wide")
st.title("🥬 Pak Choy Precise Hybrid Analyzer v4 YOLO with boundary")

# 2. Initialize Clients & ONNX Runtime Engine
@st.cache_resource
def load_onnx_session():
    # Directly loads the universal YOLOv8 matrix calculation weights file from your repository
    return ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])

try:
    session = load_onnx_session()
    input_name = session.get_inputs()[0].name
except Exception as e:
    st.error("Missing 'yolov8n.onnx' file in your main repository path! Please upload it to GitHub.")

api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    api_key = "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"
client = Groq(api_key=api_key)

def encode_image_data(file_bytes):
    return base64.b64encode(file_bytes).decode("utf-8")

# 3. Input Methods
st.subheader("📸 Choose Input Method")
source_option = st.radio(
    "Select how you want to provide images:",
    ("Upload from Photo Library", "Take Live Photo with iPhone Camera"),
)

images_to_process = []

if source_option == "Upload from Photo Library":
    uploaded_files = st.file_uploader(
        label="Upload Pak Choy Leaf Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for f in uploaded_files:
            images_to_process.append({"name": f.name, "bytes": f.read()})
else:
    camera_file = st.camera_input("Snap a photo of the leaf")
    if camera_file:
        images_to_process.append(
            {"name": "Live_Camera_Capture.jpg", "bytes": camera_file.read()}
        )

# 4. Processing Phase
if images_to_process:
    st.info(f"Processing {len(images_to_process)} image(s)...")

    for index, img_data in enumerate(images_to_process):
        st.write("---")
        col1, col2 = st.columns([1, 1])
        original_name = img_data["name"]
        original_image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        orig_w, orig_h = original_image.size
        
        with col1:
            st.markdown("### 🔍 Vision Bounding Box Output")
            
            with st.spinner("Executing YOLOv8 structural detection engine via ONNX..."):
                leaf_count = 0
                annotated_image = original_image.copy()
                draw = ImageDraw.Draw(annotated_image)
                
                try:
                    # Preprocess raw image pixels into the exact 640x640 shape YOLO expects
                    img_resized = original_image.resize((640, 640))
                    img_np = np.array(img_resized).astype(np.float32) / 255.0
                    img_np = np.transpose(img_np, (2, 0, 1))  # HWC to CHW format
                    img_np = np.expand_dims(img_np, axis=0)    # Add batch dimension
                    
                    # Run inference inside the ultra-lightweight math engine
                    outputs = session.run(None, {input_name: img_np})
                    predictions = np.squeeze(outputs[0])  # Shape: [84, 8400]
                    
                    # Separate coordinate arrays from the class prediction scores
                    boxes = predictions[:4, :].T  # [8400, 4] -> [x_center, y_center, width, height]
                    scores = predictions[4:, :].T  # [8400, 80]
                    
                    # Find maximum confidence class for each proposal box
                    max_scores = np.max(scores, axis=1)
                    class_ids = np.argmax(scores, axis=1)
                    
                    # General structural plants/vegetation indices filter
                    # Maps 58 (potted plant) and nearby organic texture categories
                    conf_threshold = 0.15
                    keep_indices = np.where((max_scores > conf_threshold) & ((class_ids == 58) | (class_ids == 9) | (class_ids == 0)))[0]
                    
                    # Gather passing box components
                    filtered_boxes = boxes[keep_indices]
                    filtered_scores = max_scores[keep_indices]
                    
                    # Apply Non-Maximum Suppression (NMS) to clear out overlapping double boxes
                    x1 = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
                    y1 = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
                    x2 = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
                    y2 = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2
                    areas = (x2 - x1) * (y2 - y1)
                    
                    order = filtered_scores.argsort()[::-1]
                    keep = []
                    while order.size > 0:
                        i = order[0]
                        keep.append(i)
                        xx1 = np.maximum(x1[i], x1[order[1:]])
                        yy1 = np.maximum(y1[i], y1[order[1:]])
                        xx2 = np.minimum(x2[i], x2[order[1:]])
                        yy2 = np.minimum(y2[i], y2[order[1:]])
                        w = np.maximum(0.0, xx2 - xx1)
                        h = np.maximum(0.0, yy2 - yy1)
                        inter = w * h
                        ovr = inter / (areas[i] + areas[order[1:]] - inter)
                        inds = np.where(ovr <= 0.45)[0]
                        order = order[inds + 1]
                    
                    # Draw final isolated bounding frames
                    for idx in keep:
                        box = filtered_boxes[idx]
                        x_center, y_center, w_box, h_box = box
                        
                        # Translate normalized 640 coordinates back into actual pixels
                        abs_x1 = int((x_center - w_box / 2) * (orig_w / 640.0))
                        abs_y1 = int((y_center - h_box / 2) * (orig_h / 640.0))
                        abs_x2 = int((x_center + w_box / 2) * (orig_w / 640.0))
                        abs_y2 = int((y_center + h_box / 2) * (orig_h / 640.0))
                        
                        leaf_count += 1
                        draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline="red", width=4)
                        draw.text((abs_x1 + 6, abs_y1 + 6), f"#{leaf_count}", fill="red")
                                
                except Exception as ex:
                    st.error(f"Vision Processing Error: {ex}")
            
            st.image(annotated_image, caption=f"YOLO ONNX Output for {original_name}", use_container_width=True)

        with col2:
            st.subheader(f"Analysis: {original_name}")
            st.metric(label="🌱 Total Target Leaves Detected", value=leaf_count)

            # --- PHASE 2: LLM PATHOLOGY ANALYSIS ---
            with st.spinner("Generating diagnostic assessment via LLM..."):
                try:
                    encoded_image = encode_image_data(img_data["bytes"])
                    response = client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": """
                                        Crop: Pak choy
                                        Task: Evaluate this image for visual plant pathology anomalies.
                                        1. Review leaf textures, margins, spotting patterns, and damage signatures.
                                        2. Construct a diagnosis listing at least 3 distinct likely root causes.
                                        
                                        Output format: Return ONLY a raw valid JSON object. Do not include markdown codeblocks or backticks.
                                        
                                        JSON Template structure:
                                        {
                                            "visible_symptoms": "Summary of overall observed plant symptoms",
                                            "possible_causes": "List at least 3 distinct potential causes or diseases",
                                            "recommended_intervention": "Action steps to control and treat the problem"
                                        }
                                        """,
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{encoded_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        temperature=0.1,
                        max_completion_tokens=600,
                    )

                    raw_content = response.choices[0].message.content.strip()
                    if "```" in raw_content:
                        raw_content = raw_content.split("```json")[-1].split("```")[0].strip()

                    analysis_data = json.loads(raw_content, strict=False)

                    st.markdown("### 🩺 Diagnostic Findings")
                    diagnostic_table = {
                        "Analysis Field": [
                            "Observed Symptoms Summary",
                            "Possible Root Causes (At least 3)",
                            "Recommended Management Plan",
                        ],
                        "Findings": [
                            analysis_data.get("visible_symptoms", "N/A"),
                            analysis_data.get("possible_causes", "N/A"),
                            analysis_data.get("recommended_intervention", "N/A"),
                        ],
                    }
                    st.table(diagnostic_table)

                except Exception as e:
                    st.error(f"Error parsing analysis: {e}")
else:
    st.info("Provide an image using one of the methods above to generate advice.")
