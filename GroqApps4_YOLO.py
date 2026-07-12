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
st.title("🥬 Pak Choy Pure Local Hybrid Analyzer v4 with YOLO")

# 2. Initialize Lightweight Inference Engine
@st.cache_resource
def load_onnx_session():
    # Looks for the local ONNX weights file pushed to your repository
    # Bypasses torch and heavy ML libraries completely
    return ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])

try:
    session = load_onnx_session()
    input_name = session.get_inputs()[0].name
except Exception as e:
    st.error("Missing 'yolov8n.onnx' model file in your main repository path!")

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
            st.markdown("### 🔍 Local Bounding Box Output")
            
            with st.spinner("Calculating mathematical bounding matrices via ONNX..."):
                try:
                    # Preprocess image to fit YOLO standard inputs (640x640 float matrix)
                    img_resized = original_image.resize((640, 640))
                    img_array = np.array(img_resized).astype(np.float32) / 255.0
                    img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
                    img_array = np.expand_dims(img_array, axis=0)   # Add batch dim

                    # Run inference via the tiny native engine
                    outputs = session.run(None, {input_name: img_array})
                    predictions = np.squeeze(outputs[0])
                    
                    # Parse bounding predictions 
                    boxes = predictions[:4, :].T
                    scores = np.max(predictions[4:, :], axis=0)
                    class_ids = np.argmax(predictions[4:, :], axis=0)

                    # Filter for object thresholds (Focusing on plant/vegetation classes)
                    # Class index 58 is typically 'potted plant' in standard weights
                    mask = (scores > 0.25) & ((class_ids == 58) | (class_ids == 0))
                    valid_boxes = boxes[mask]
                    valid_scores = scores[mask]

                    yolo_count = len(valid_boxes)

                    annotated_image = original_image.copy()
                    draw = ImageDraw.Draw(annotated_image)

                    for idx, box in enumerate(valid_boxes):
                        x_center, y_center, w, h = box
                        
                        # Rescale bounding box coordinates back to original image size
                        x1 = int((x_center - w / 2) * (orig_w / 640.0))
                        y1 = int((y_center - h / 2) * (orig_h / 640.0))
                        x2 = int((x_center + w / 2) * (orig_w / 640.0))
                        y2 = int((y_center + h / 2) * (orig_h / 640.0))

                        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                        draw.text((x1 + 6, y1 + 6), f"#{idx+1}", fill="red")

                    st.image(annotated_image, caption=f"Local Output for {original_name}", use_container_width=True)
                
                except Exception as ex:
                    st.error(f"Localized Object Mapping Error: {ex}")
                    yolo_count = 0
                    st.image(original_image, caption=original_name, use_container_width=True)

        with col2:
            st.subheader(f"Analysis: {original_name}")
            st.metric(label="🌱 Total Target Leaves Detected", value=yolo_count)

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
