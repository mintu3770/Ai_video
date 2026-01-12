import streamlit as st
import google.generativeai as genai
from huggingface_hub import InferenceClient
import io
from PIL import Image

# --- Configuration & Secrets ---
# For local testing, create a .streamlit/secrets.toml file with:
# GOOGLE_API_KEY = "your_key"
# HF_TOKEN = "your_hf_key"

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

if "HF_TOKEN" in st.secrets:
    # We use the free InferenceClient
    hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
else:
    st.error("Missing Hugging Face Token in secrets!")
    st.stop()

# --- Functions ---

def generate_catchy_phrase(prompt):
    """Generates text using Gemini (Free Tier)."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        f"Write a single, short, punchy, and viral social media caption (under 15 words) for a video about: {prompt}. No hashtags, just the phrase."
    )
    return response.text.strip()

def generate_poster(prompt):
    """Generates an image using Flux.1-dev via Hugging Face (Free)."""
    # FLUX.1-dev is currently one of the best open models
    image = hf_client.text_to_image(
        f"Movie poster for {prompt}, cinematic, 8k, typography, title text",
        model="black-forest-labs/FLUX.1-dev"
    )
    return image

def generate_video(prompt):
    """
    Generates a video using the Damo-Vilab model via Hugging Face.
    Note: Free tier video models are small (low res) and short (2s),
    but they are completely free!
    """
    # This specific call uses the text-to-video-ms-1.7b model
    # We output bytes directly
    video_bytes = hf_client.text_to_video(
        prompt,
        model="damo-vilab/text-to-video-ms-1.7b"
    )
    return video_bytes

# --- Streamlit UI ---

st.set_page_config(page_title="Free AI Content Tool", page_icon="🎓")

st.title("🎓 The Student's AI Studio")
st.markdown("Generate content for **$0** using Gemini & Hugging Face.")

user_prompt = st.text_input("Enter your content idea:", placeholder="e.g., A robot painting a canvas in space")

if st.button("Generate for Free"):
    if not user_prompt:
        st.warning("Please enter a prompt first!")
    else:
        st.info("🚀 Creating content (this might take 30s as we use free servers)...")
        
        col1, col2 = st.columns([1, 1])

        # 1. Catchy Phrase (Gemini)
        with st.spinner("Gemini is thinking..."):
            try:
                phrase = generate_catchy_phrase(user_prompt)
                st.success("Caption Ready!")
                st.markdown(f"### 📢 {phrase}")
            except Exception as e:
                st.error(f"Text Error: {e}")

        # 2. Poster (Hugging Face Flux)
        with st.spinner("Generating Poster (Flux)..."):
            try:
                # Flux takes about 10-15 seconds on free tier
                poster_image = generate_poster(user_prompt)
                with col1:
                    st.image(poster_image, caption="Flux.1 Poster", use_container_width=True)
            except Exception as e:
                st.warning("Flux is busy (common on free tier). Trying SDXL...")
                try:
                    # Fallback to Stable Diffusion XL if Flux is busy
                    poster_image = hf_client.text_to_image(
                        f"Movie poster for {prompt}, cinematic",
                        model="stabilityai/stable-diffusion-xl-base-1.0"
                    )
                    with col1:
                        st.image(poster_image, caption="SDXL Poster", use_container_width=True)
                except Exception as inner_e:
                    st.error(f"Image Error: {inner_e}")

        # 3. Video (Hugging Face Video)
        with st.spinner("Generating Video (Standard Free Model)..."):
            try:
                video_data = generate_video(user_prompt)
                
                # Save bytes to a temporary file so Streamlit can read it
                # (Hugging Face returns raw bytes)
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(video_data)
                    tfile_path = tfile.name
                
                with col2:
                    st.video(tfile_path)
            except Exception as e:
                st.error(f"Video Error (Free servers might be overloaded): {e}")
