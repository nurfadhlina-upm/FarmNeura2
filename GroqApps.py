import base64
from groq import Groq
import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Crop Disease Analyzer", layout="wide")
st.title("🥬 Pak Choy Health Analyzer")

# 2. Safely Fetch Key from Streamlit Secrets
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    # Fallback to the working key you provided if secrets aren't set up yet
    api_key = "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"

client = Groq(api_key=api_key)


def encode_image_data(file_bytes):
    return base64.b64encode(file_bytes).decode("utf-8")


# 3. Double input options for Mobile Devices
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
    # st.camera_input automatically requests permissions to launch the iPhone camera inside Safari/Chrome
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

        with col1:
            st.image(img_data["bytes"], caption=img_data["name"])

        with col2:
            st.subheader(f"Analysis: {img_data['name']}")

            with st.spinner("Analyzing with Llama 4 Scout..."):
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

                                        Analyse the image. Return:
                                        1. Visible symptoms
                                        2. Possible causes
                                        3. Additional checks required
                                        4. Recommended intervention
                                        5. Confidence and limitations
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
                        temperature=0.2,
                        max_completion_tokens=700,
                    )

                    st.markdown(response.choices[0].message.content)

                except Exception as e:
                    st.error(f"Error analyzing image: {e}")
else:
    st.info("Provide an image using one of the methods above to generate advice.")