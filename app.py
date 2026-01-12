import streamlit as st
import requests
from huggingface_hub import InferenceClient
import tempfile

# --- Configuration & Secrets ---
# Ensure you have GOOGLE_API_KEY and HF_TOKEN in .streamlit/secrets.toml
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing GOOGLE_API_KEY in secrets.")
    st.stop()

if "HF_TOKEN" not in st.secrets:
    st.error("Missing HF_TOKEN in secrets.")
    st.stop()

hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])

# --- Functions ---

def generate_gemini_text(prompt):
    """
    Connects to Google's API using the latest 2026 models.
    """
    api_key = st.secrets["GOOGLE_API_KEY"]
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": f"Write a single, short, punchy, viral social media caption (under 15 words) for a video about: {prompt}. No hashtags, just the phrase."}]
        }]
    }

    # Attempt 1: Gemini 2.5 Flash (Current Standard for 2026)
    #
    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            # Attempt 2: Gemini 2.0 Flash (Fallback)
            print(f"2.5 Flash failed ({response.status_code}), trying 2.0...")
            fallback_model = "gemini-2.0-flash"
            url_fallback = f"https://generativelanguage.googleapis.com/v1beta/models/{fallback_model}:generateContent?key={api_key}"
            
            response_fallback = requests.post(url_fallback, headers=headers, json=payload)
            if response_fallback.status_code == 200:
                return response_fallback.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                return f"Error: {response.text}"
                
    except Exception as e:
        return f"Connection Error: {e}"

def generate_poster(prompt):
    """Generates an image using Stable Diffusion v1.5 (Most Reliable Free Model)."""
    try:
        # We use v1-5 because it is never gated and rarely hits payment limits.
        #
        image = hf_client.text_to_image(
            f"Movie poster for {prompt}, cinematic, 8k, typography, title text",
            model="runwayml/stable-diffusion-v1-5" 
        )
        return image, "SD v1.5"
    except Exception as e:
        return None, f"Image Error: {e}"

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
st.markdown("Text by **Gemini 2.5** | Visuals by **Stable Diffusion v1.5**")

user_prompt = st.text_input("Enter content idea:", placeholder="e.g., A futuristic samurai in a neon city")

if st.button("Generate Content"):
    if not user_prompt:
        st.warning("Please enter a prompt first!")
    else:
        st.info("🚀 Launching Hybrid Agents...")
        
        col1, col2 = st.columns(2)

        # 1. TEXT (Gemini 2.5)
        with st.spinner("Gemini is writing..."):
            caption = generate_gemini_text(user_prompt)
            if "Error" in caption:
                st.error("Text Generation Failed")
                st.code(caption) 
            else:
                st.success("✅ Caption Ready")
                st.markdown(f"### 📢 {caption}")

        # 2. IMAGE (Hugging Face)
        with st.spinner("Drawing Poster..."):
            img, model_name = generate_poster(user_prompt)
            if img:
                with col1:
                    st.image(img, caption=f"Poster ({model_name})", use_container_width=True)
                st.success("✅ Poster Ready")
            else:
                st.error(model_name) # Prints the error message

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
