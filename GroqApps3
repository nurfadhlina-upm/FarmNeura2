import base64
import json
from groq import Groq
import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Crop Disease Analyzer", layout="wide")
st.title("🥬 Pak Choy Health Analyzer & Leaf Directory 3")

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
        col1, col2 = st.columns([1, 2])

        # Fixed Line 56 syntax error by breaking it into two explicit steps
        original_name = img_data["name"]

        with col1:
            st.image(img_data["bytes"], caption=original_name)

        with col2:
            st.subheader(f"Analysis: {original_name}")

            with st.spinner("Compiling Leaf Directory with Llama 4 Scout..."):
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
                                        Days after planting: 25

                                        Task: Perform a systematic audit of the leaves in the image.
                                        1. Count all individual visible leaves systematically.
                                        2. Build a segmented descriptive list mapping out each leaf by location (e.g., "Leaf 1: Lower right outer leaf", "Leaf 2: Central new shoot") and describe its specific health status.
                                        3. Detail at least 3 separate possible root causes for any issues.

                                        Output format: Return ONLY a raw valid JSON object. Do not include markdown codeblocks or backticks.

                                        JSON Template structure:
                                        {
                                            "leaf_count": 0,
                                            "leaf_directory": [
                                                {"leaf_id": "Leaf 1", "location": "Description of position on plant", "condition": "Healthy / Holes / Yellow spots"},
                                                {"leaf_id": "Leaf 2", "location": "Description of position on plant", "condition": "Healthy / Holes / Yellow spots"}
                                            ],
                                            "visible_symptoms": "Overall summary of spots, holes, or discoloration observed",
                                            "possible_causes": "Thorough breakdown listing at least 3 distinct likely root causes or diseases",
                                            "additional_checks": "Specific steps or tests the farmer should perform to confirm the diagnosis",
                                            "recommended_intervention": "Action steps to control and treat the problem",
                                            "confidence_level": "High/Medium/Low"
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
                        max_completion_tokens=1000,
                    )

                    raw_content = response.choices[0].message.content.strip()

                    if raw_content.startswith("```"):
                        raw_content = (
                            raw_content.replace("```json", "")
                            .replace("```", "")
                            .strip()
                        )

                    analysis_data = json.loads(raw_content)

                    # 1. Total Count Metric Banner
                    st.metric(
                        label="🌱 Total Audited Leaves",
                        value=analysis_data.get("leaf_count", 0),
                    )

                    # 2. Segmented Leaf Directory Table
                    st.markdown("### 📊 Segmented Leaf Directory")
                    directory_data = analysis_data.get("leaf_directory", [])

                    if directory_data:
                        directory_table = {
                            "Leaf ID": [item.get("leaf_id", "N/A") for item in directory_data],
                            "Location / Position": [item.get("location", "N/A") for item in directory_data],
                            "Health Status": [item.get("condition", "N/A") for item in directory_data]
                        }
                        st.table(directory_table)
                    else:
                        st.warning("No individual leaf segments mapped.")

                    # 3. Overall Diagnostic Breakdown Table
                    st.markdown("### 🩺 Diagnostic Findings")
                    table_rows = {
                        "Analysis Field": [
                            "Visible Symptoms Summary",
                            "Possible Causes (Differential Diagnosis)",
                            "Additional Checks Required",
                            "Recommended Intervention",
                            "Confidence Level",
                        ],
                        "AI Diagnostic Findings": [
                            analysis_data.get("visible_symptoms", "N/A"),
                            analysis_data.get("possible_causes", "N/A"),
                            analysis_data.get("additional_checks", "N/A"),
                            analysis_data.get("recommended_intervention", "N/A"),
                            analysis_data.get("confidence_level", "N/A"),
                        ],
                    }
                    st.table(table_rows)

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
