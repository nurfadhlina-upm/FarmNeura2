import base64
import io
import json
from groq import Groq
from PIL import Image
import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Crop Disease Analyzer", layout="wide")
st.title("🥬 Pak Choy Leaf Counter & Health Analyzer v3 LLM no boundary")

# 2. Initialize Clients
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
        
        with col1:
            st.markdown("### 🔍 Uploaded Crop Target")
            st.image(original_image, caption=original_name, use_container_width=True)

        with col2:
            st.subheader(f"Analysis: {original_name}")

            # --- MULTIMODAL PATHOLOGY & COUNTING ANALYSIS ---
            with st.spinner("Analyzing plant structure and counting leaves..."):
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
                                        Task: Evaluate this image for structural leaf count and visual plant pathology anomalies.
                                        1. Systematically count the total number of distinct visible leaves on the plant structure.
                                        2. Review leaf textures, margins, spotting patterns, and damage signatures.
                                        3. Construct a diagnosis listing at least 3 distinct likely root causes.
                                        
                                        Output format: Return ONLY a raw valid JSON object. Do not include markdown codeblocks or backticks.
                                        
                                        JSON Template structure:
                                        {
                                            "total_leaf_count": 0,
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
                    leaf_count = analysis_data.get("total_leaf_count", 0)

                    # Display the leaf count visually via a metric block
                    st.metric(label="🌱 Total Leaves Counted", value=leaf_count)

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
                    st.code(raw_content if 'raw_content' in locals() else "No data received.")
else:
    st.info("Provide an image using one of the methods above to generate advice.")
