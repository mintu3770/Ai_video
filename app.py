import streamlit as st
import requests
from huggingface_hub import InferenceClient
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

def generate_gemini_text(prompt):
    """
    Connects directly to Gemini 1.5 Flash via REST API.
    This bypasses the 'Model not found' library errors.
    """
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    # 1. Try Gemini 1.5 Flash (Newest/Fastest)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": f"Write a single, short, punchy, viral social media caption (under 15 words) for a video about: {prompt}. No hashtags, just the phrase."}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            # If 1.5 Flash fails (e.g., 404), try the older Gemini Pro
            print(f"Flash failed ({response.status_code}), switching to Pro...")
            url_fallback = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
            response_fallback = requests.post(url_fallback, headers=headers, json=payload)
            
            if response_fallback.status_code == 200:
                return response_fallback.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                return f"Error: {response.text}"
                
    except Exception as e:
        return f"Connection Error: {e}"

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

        # 1. TEXT (Gemini via Direct Web Call)
        with st.spinner("Gemini is writing..."):
            caption = generate_gemini_text(user_prompt)
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
