import streamlit as st
import google.generativeai as genai
from huggingface_hub import InferenceClient
import io
from PIL import Image
import tempfile

# --- Configuration & Secrets ---
# Ensure you have set GOOGLE_API_KEY and HF_TOKEN in .streamlit/secrets.toml
# or in your deployment settings.

# 1. Authenticate Gemini
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Missing GOOGLE_API_KEY. Please add it to your secrets.")
    st.stop()

# 2. Authenticate Hugging Face
if "HF_TOKEN" in st.secrets:
    hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
else:
    st.error("Missing HF_TOKEN. Please add it to your secrets.")
    st.stop()

# --- Functions ---

def generate_catchy_phrase(prompt):
    """Generates text using Gemini 1.5 Flash."""
    try:
        # Using the standard model name. Ensure google-generativeai >= 0.8.3
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            f"Write a single, short, punchy, and viral social media caption (under 15 words) for a video about: {prompt}. No hashtags, just the phrase."
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating text: {e}"

def generate_poster(prompt):
    """Generates an image using Flux.1-dev with fallback to SDXL."""
    # Attempt 1: Flux (High Quality, but often busy)
    try:
        image = hf_client.text_to_image(
            f"Movie poster for {prompt}, cinematic, 8k, typography, title text",
            model="black-forest-labs/FLUX.1-dev"
        )
        return image, "Flux.1-dev"
    except Exception:
        # Attempt 2: Stable Diffusion XL (Reliable fallback)
        print("Flux busy, switching to SDXL...")
        image = hf_client.text_to_image(
            f"Movie poster for {prompt}, cinematic",
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )
        return image, "SDXL-Base"

def generate_video(prompt):
    """
    Generates a video using Damo-Vilab.
    WARNING: Free tier video models often timeout (503 Service Unavailable).
    """
    video_bytes = hf_client.text_to_video(
        prompt,
        model="damo-vilab/text-to-video-ms-1.7b"
    )
    return video_bytes

# --- Streamlit UI ---

st.set_page_config(page_title="Free AI Content Tool", page_icon="🎓")

st.title("🎓 The Student's AI Studio")
st.markdown("Generate content for **$0** using Gemini & Hugging Face.")

# Input
user_prompt = st.text_input("Enter your content idea:", placeholder="e.g., A robot painting a canvas in space")

if st.button("Generate for Free"):
    if not user_prompt:
        st.warning("Please enter a prompt first!")
    else:
        st.info("🚀 Creating content (Images take ~10s, Video takes ~30s)...")
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])

        # 1. Text Generation (Gemini)
        with st.spinner("Gemini is thinking..."):
            phrase = generate_catchy_phrase(user_prompt)
            st.success("Caption Ready!")
            st.markdown(f"### 📢 {phrase}")

        # 2. Image Generation (Flux/SDXL)
        with st.spinner("Generating Poster..."):
            try:
                # We pass 'user_prompt' directly to avoid NameError
                poster_image, model_used = generate_poster(user_prompt)
                
                with col1:
                    st.image(poster_image, caption=f"Poster ({model_used})", use_container_width=True)
            except Exception as e:
                with col1:
                    st.error(f"Image generation failed: {e}")

        # 3. Video Generation (Experimental)
        with st.spinner("Generating Video (May fail on free tier)..."):
            try:
                video_data = generate_video(user_prompt)
                
                # Save to temp file for display
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(video_data)
                    tfile_path = tfile.name
                
                with col2:
                    st.video(tfile_path)
            except Exception as e:
                with col2:
                    st.warning(f"Video generation skipped (Server Busy/Timeout). This is common on the free tier.\nError details: {e}")
