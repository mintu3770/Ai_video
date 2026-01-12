import streamlit as st
import requests
from huggingface_hub import InferenceClient
import io
from PIL import Image
import tempfile

# --- Configuration & Secrets ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing GOOGLE_API_KEY in secrets.")
    st.stop()

if "HF_TOKEN" not in st.secrets:
    st.error("Missing HF_TOKEN in secrets.")
    st.stop()

# Initialize Hugging Face Client
hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])

# --- Functions ---

def generate_text_gemini(prompt):
    """
    Direct Web Request to Gemini 1.5 Flash.
    Bypasses the 'library version' errors on Streamlit Cloud.
    """
    api_key = st.secrets["GOOGLE_API_KEY"]
    # We use the REST API URL directly
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": f"Write a single, short, viral social media caption (under 15 words) for a video about: {prompt}. No hashtags."}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            return f"Gemini Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"Connection Error: {e}"

def generate_image_hf(prompt):
    """
    Uses Stable Diffusion v1.5.
    Why? It is the only model that is 100% free and ungated.
    Models like Flux (Payment Required) or SD 2.1 (Repo Not Found) are avoided.
    """
    try:
        image = hf_client.text_to_image(
            f"cinematic movie poster for {prompt}, high quality",
            model="runwayml/stable-diffusion-v1-5"
        )
        return image
    except Exception as e:
        st.error(f"Image Error: {e}")
        return None

def generate_video_hf(prompt):
    """
    Experimental Video Generation.
    Note: Free tier video servers are often overloaded (503 error).
    """
    try:
        video_bytes = hf_client.text_to_video(
            prompt,
            model="damo-vilab/text-to-video-ms-1.7b"
        )
        return video_bytes
    except Exception:
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="AI Studio (Fixed)", page_icon="🛠️")

st.title("🛠️ AI Studio: The 'Reliable' Build")
st.markdown("Text via **Gemini REST** | Images via **Stable Diffusion v1.5**")

user_prompt = st.text_input("Enter content idea:", placeholder="e.g., A robot painting in the rain")

if st.button("Generate Content"):
    if not user_prompt:
        st.warning("Please enter a prompt!")
    else:
        col1, col2 = st.columns(2)

        # 1. TEXT (Gemini Direct)
        with st.spinner("Gemini is thinking..."):
            caption = generate_text_gemini(user_prompt)
            if "Error" in caption:
                st.error(caption)
            else:
                st.success("✅ Caption Ready")
                st.markdown(f"### 📢 {caption}")

        # 2. IMAGE (SD v1.5)
        with st.spinner("Generating Poster (SD v1.5)..."):
            img = generate_image_hf(user_prompt)
            if img:
                with col1:
                    st.image(img, caption="Generated Poster", use_container_width=True)
                st.success("✅ Poster Ready")
            else:
                st.error("Image generation failed.")

        # 3. VIDEO (Experimental)
        with st.spinner("Attempting Video (May fail)..."):
            vid_bytes = generate_video_hf(user_prompt)
            if vid_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(vid_bytes)
                    vid_path = tfile.name
                with col2:
                    st.video(vid_path)
                st.success("✅ Video Ready")
            else:
                with col2:
                    st.warning("Video Server Busy (Common on Free Tier).")
