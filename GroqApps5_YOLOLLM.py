import base64
import io
import json
from groq import Groq
from PIL import Image, ImageDraw
import requests
import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Crop Disease Analyzer", layout="wide")
st.title("🥬 Pak Choy Hybrid Detection & Health Analyzer")

# 2. Safely Fetch Keys from Streamlit Secrets
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    api_key = "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"

roboflow_key = st.secrets.get("ROBOFLOW_API_KEY")

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

        # Convert raw binary bytes into a readable PIL container image
        original_image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")

        with col1:
            st.markdown("### 🔍 Cloud Bounding Box Output")

            if not roboflow_key:
                st.warning(
                    "Please add ROBOFLOW_API_KEY to your Streamlit Secrets."
                )
                yolo_count = 0
                st.image(
                    original_image,
                    caption=original_name,
                    use_container_width=True,
                )
            else:
                with st.spinner("Running high-precision leaf localization..."):
                    try:
                        # Call Roboflow hosted inference engine (bypasses heavy server memory requirements)
                        # Replace "microsoft-coco-bounding-boxes/1" with a custom leaf model ID if preferred
                        upload_url = f"https://detect.roboflow.com/microsoft-coco-bounding-boxes/1?api_key={roboflow_key}"

                        response_rf = requests.post(
                            upload_url,
                            files={"file": io.BytesIO(img_data["bytes"])},
                        )
                        predictions = response_rf.json().get(
                            "predictions", []
                        )

                        # Filter specifically for green plant items or standard leaves
                        leaf_predictions = [
                            p for p in predictions if p["class"] in ["potted plant", "plant", "leaf", "vegetable"]
                        ]
                        
                        # Fallback: if no specific plant tag matches, use all detected objects
                        if not leaf_predictions:
                            leaf_predictions = predictions

                        yolo_count = len(leaf_predictions)

                        # Draw the crisp bounding layers
                        annotated_image = original_image.copy()
                        draw = ImageDraw.Draw(annotated_image)

                        for idx, pred in enumerate(leaf_predictions):
                            x_center = pred["x"]
                            y_center = pred["y"]
                            width = pred["width"]
                            height = pred["height"]

                            # Calculate absolute coordinates
                            x1 = int(x_center - (width / 2))
                            y1 = int(y_center - (height / 2))
                            x2 = int(x_center + (width / 2))
                            y2 = int(y_center + (height / 2))

                            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                            draw.text((x1 + 6, y1 + 6), f"#{idx+1}", fill="red")

                        st.image(
                            annotated_image,
                            caption=f"Target Segmentation for {original_name}",
                            use_container_width=True,
                        )

                    except Exception as e:
                        st.error(f"Cloud Object Detection Error: {e}")
                        yolo_count = 0
                        st.image(
                            original_image,
                            caption=original_name,
                            use_container_width=True,
                        )

        with col2:
            st.subheader(f"Analysis: {original_name}")

            # Display the leaf counts calculated entirely by the Computer Vision network
            st.metric(label="🌱 Total Leaves Tracked (via YOLO)", value=yolo_count)

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
                        raw_content = (
                            raw_content.split("```json")[-1]
                            .split("```")[0]
                            .strip()
                        )

                    analysis_data = json.loads(raw_content)

                    # Render pathology matrix cleanly
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
                    st.text("Raw response was:")
                    st.code(
                        response.choices[0].message.content
                        if "response" in locals()
                        else "No response received."
                    )
else:
    st.info("Provide an image using one of the methods above to generate advice.")
