import base64
import io
import json
from groq import Groq
from PIL import Image, ImageDraw
import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Crop Disease Analyzer", layout="wide")
st.title("🥬 Pak Choy Leaf Detector & Segmenter")

# 2. Safely Fetch Key from Streamlit Secrets
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

        original_image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
        img_width, img_height = original_image.size
        original_name = img_data["name"]

        with col2:
            st.subheader(f"Analysis: {original_name}")

            with st.spinner("Locating leaves and performing health diagnostic..."):
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
                                        Task: Perform structural detection and health breakdown.
                                        1. Detect ALL visible pak choy leaves. You must provide a bounding box entry for every single individual leaf counted.
                                        2. Identify visible global symptoms and list at least 3 distinct possible causes (diseases, pests, or deficiencies).
                                        
                                        Output format: Return ONLY a raw valid JSON object. Do not include markdown codeblocks or backticks.
                                        
                                        JSON Template structure:
                                        {
                                            "leaf_boxes": [[ymin, xmin, ymax, xmax], [ymin, xmin, ymax, xmax]],
                                            "visible_symptoms": "Summary of overall observed symptoms",
                                            "possible_causes": "List at least 3 distinct potential causes or diseases",
                                            "recommended_intervention": "Action steps to take"
                                        }
                                        
                                        Note: Coordinates must be integers normalized on a 0 to 1000 scale relative to the image edges.
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
                        max_completion_tokens=1000,
                    )

                    raw_content = response.choices[0].message.content.strip()

                    # Clean markdown wrappers if present
                    if "```" in raw_content:
                        raw_content = raw_content.split("```json")[-1].split("```")[0].strip()

                    analysis_data = json.loads(raw_content)
                    
                    # Robust multi-key check to make sure boxes never fail to read
                    leaf_boxes = analysis_data.get("leaf_boxes") or analysis_data.get("boxes") or []
                    total_count = len(leaf_boxes)

                    # --- PYTHON DRAWING ENGINE ---
                    annotated_image = original_image.copy()
                    draw = ImageDraw.Draw(annotated_image)

                    for idx, box in enumerate(leaf_boxes):
                        if len(box) == 4:
                            ymin, xmin, ymax, xmax = box
                            
                            # Scale back to local image pixel size
                            abs_xmin = int((xmin / 1000.0) * img_width)
                            abs_ymin = int((ymin / 1000.0) * img_height)
                            abs_xmax = int((xmax / 1000.0) * img_width)
                            abs_ymax = int((ymax / 1000.0) * img_height)

                            # Clean drawing implementation
                            draw.rectangle(
                                [abs_xmin, abs_ymin, abs_xmax, abs_ymax],
                                outline="red",
                                width=3,
                            )
                            draw.text((abs_xmin + 4, abs_ymin + 4), f"#{idx+1}", fill="red")

                    # Render the marked image inside Column 1
                    with col1:
                        st.image(
                            annotated_image,
                            caption=f"Segmented Output for {original_name}",
                            use_container_width=True
                        )

                    # Render targeted counts inside Column 2
                    st.metric(label="🌱 Total Leaves Tracked & Boxed", value=total_count)

                    # Render Pathology Table (Bounding Box data coordinates table completely removed)
                    st.markdown("### 🩺 Diagnostic Audit Summary")
                    diagnostic_table = {
                        "Analysis Field": [
                            "Observed Symptoms Summary",
                            "Possible Root Causes (At least 3)",
                            "Recommended Management Plan"
                        ],
                        "Findings": [
                            analysis_data.get("visible_symptoms", "N/A"),
                            analysis_data.get("possible_causes", "N/A"),
                            analysis_data.get("recommended_intervention", "N/A")
                        ]
                    }
                    st.table(diagnostic_table)

                except Exception as e:
                    with col1:
                        st.image(original_image, caption=original_name, use_container_width=True)
                    st.error(f"Error parsing analysis: {e}")
                    st.text("Raw response was:")
                    st.code(
                        response.choices[0].message.content
                        if "response" in locals()
                        else "No response received."
                    )
else:
    st.info("Provide an image using one of the methods above to generate advice.")
