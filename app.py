import streamlit as st
import google.generativeai as genai
from huggingface_hub import InferenceClient
import tempfile

# --- Configuration & Secrets ---
# 1. Google Auth
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Missing GOOGLE_API_KEY in secrets.")
    st.stop()

# 2. Hugging Face Auth
if "HF_TOKEN" in st.secrets:
    hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
else:
    st.error("Missing HF_TOKEN in secrets.")
    st.stop()

# --- Functions ---

def generate_catchy_phrase(prompt):
    """Generates text using Google Gemini 1.5 Flash."""
    try:
        # Using 1.5 Flash for speed and quality
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            f"Write a single, short, punchy, viral social media caption (under 15 words) for a video about: {prompt}. No hashtags, just the phrase."
        )
        return response.text.strip()
    except Exception as e:
        # Fallback to older model if Flash fails
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(f"Write a short caption for: {prompt}")
            return response.text.strip()
        except:
            return f"Gemini Error: {e}"

def generate_poster(prompt):
    """Generates an image using Flux.1-dev with fallback to SDXL."""
    try:
        # Attempt 1: Flux (Best Quality)
        image = hf_client.text_to_image(
            f"Movie poster for {prompt}, cinematic, 8k, typography, title text",
            model="black-forest-labs/FLUX.1-dev"
        )
        return image, "Flux"
    except Exception:
        # Attempt 2: SDXL (Reliable)
        print("Flux busy, switching to SDXL...")
        image = hf_client.text_to_image(
            f"Movie poster for {prompt}, cinematic",
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )
        return image, "SDXL"

def generate_video(prompt):
    """Generates a video using Damo-Vilab."""
    # Note: This is the most unstable part on the free tier
    video_bytes = hf_client.text_to_video(
        prompt,
        model="damo-vilab/text-to-video-ms-1.7b"
    )
    return video_bytes

# --- Streamlit UI ---

st.set_page_config(page_title="Hybrid AI Studio", page_icon="🦄")

st.title("🦄 The Hybrid AI Studio")
st.markdown("Text by **Gemini** | Visuals by **Hugging Face**")

user_prompt = st.text_input("Enter content idea:", placeholder="e.g., A futuristic samurai in a neon city")

if st.button("Generate Content"):
    if not user_prompt:
        st.warning("Please enter a prompt first!")
    else:
        st.info("🚀 Launching Hybrid Agents...")
        
        col1, col2 = st.columns(2)

        # 1. TEXT (Gemini)
        with st.spinner("Gemini is writing..."):
            caption = generate_catchy_phrase(user_prompt)
            if "Error" in caption:
                st.error(caption)
            else:
                st.success("✅ Caption Ready")
                st.markdown(f"### 📢 {caption}")

        # 2. IMAGE (Hugging Face)
        with st.spinner("Drawing Poster..."):
            try:
                img, model_name = generate_poster(user_prompt)
                with col1:
                    st.image(img, caption=f"Poster ({model_name})", use_container_width=True)
                st.success("✅ Poster Ready")
            except Exception as e:
                st.error(f"Image Failed: {e}")

        # 3. VIDEO (Hugging Face)
        with st.spinner("Rendering Video (May timeout)..."):
            try:
                vid_bytes = generate_video(user_prompt)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(vid_bytes)
                    vid_path = tfile.name
                with col2:
                    st.video(vid_path)
                st.success("✅ Video Ready")
            except Exception as e:
                with col2:
                    st.warning("Video skipped (Server Busy).")
