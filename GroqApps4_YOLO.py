import base64
import io
import json
from groq import Groq
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Crop Disease Analyzer", layout="wide")
st.title("🥬 Pak Choy Precise Hybrid Analyzer")

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
            st.markdown("### 🔍 Vision Bounding Box Output")
            
            with st.spinner("Extracting leaf clusters via excess green analysis..."):
                leaf_count = 0
                annotated_image = original_image.copy()
                draw = ImageDraw.Draw(annotated_image)
                
                try:
                    # Convert to NumPy array for fast calculations
                    img_np = np.array(original_image)
                    h, w, _ = img_np.shape
                    
                    # Compute Excess Green Index (ExG = 2G - R - B) to isolate vibrant crop leaves
                    r = img_np[:, :, 0].astype(float)
                    g = img_np[:, :, 1].astype(float)
                    b = img_np[:, :, 2].astype(float)
                    exg = 2 * g - r - b
                    
                    # Create binary mask (True where pixels are strongly green)
                    binary_mask = exg > 35
                    
                    # --- LIGHTWEIGHT NATIVE CLUSTERING (No Scipy Required) ---
                    # Downsample slightly to accelerate calculations on large mobile photos
                    scale = 4
                    small_h, small_w = h // scale, w // scale
                    
                    # Resize binary mask using simple striding
                    mask_small = binary_mask[::scale, ::scale]
                    visited = np.zeros_like(mask_small, dtype=bool)
                    
                    # Flood-fill scan to trace separate leaf targets
                    for y in range(small_h):
                        for x in range(small_w):
                            if mask_small[y, x] and not visited[y, x]:
                                # Found a new unmapped leaf component!
                                queue = [(y, x)]
                                visited[y, x] = True
                                
                                ymin, ymax = y, y
                                xmin, xmax = x, x
                                size = 0
                                
                                # Process cluster pixels
                                while queue:
                                    cy, cx = queue.pop(0)
                                    size += 1
                                    
                                    ymin, ymax = min(ymin, cy), max(ymax, cy)
                                    xmin, xmax = min(xmin, cx), max(xmax, cx)
                                    
                                    # Inspect 4-way neighbors
                                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                        ny, nx = cy + dy, cx + dx
                                        if 0 <= ny < small_h and 0 <= nx < small_w:
                                            if mask_small[ny, nx] and not visited[ny, nx]:
                                                visited[ny, nx] = True
                                                queue.append((ny, nx))
                                
                                # Filter out small background weed spots or isolated noise
                                if size > 40:
                                    leaf_count += 1
                                    # Scale bounding area coordinates back up to full image pixels
                                    abs_ymin, abs_ymax = ymin * scale, ymax * scale
                                    abs_xmin, abs_xmax = xmin * scale, xmax * scale
                                    
                                    draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline="red", width=4)
                                    draw.text((abs_xmin + 6, abs_ymin + 6), f"#{leaf_count}", fill="red")
                                    
                except Exception as ex:
                    st.error(f"Vision Processing Error: {ex}")
            
            # Show the generated visual output mapping tracking boxes cleanly
            st.image(annotated_image, caption=f"Local Output for {original_name}", use_container_width=True)

        with col2:
            st.subheader(f"Analysis: {original_name}")
            st.metric(label="🌱 Total Target Leaves Segmented", value=leaf_count)

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
                    st.text("Raw response was:")
                    st.code(raw_content if 'raw_content' in locals() else "No data received.")
else:
    st.info("Provide an image using one of the methods above to generate advice.")
