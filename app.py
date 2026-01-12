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

hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])

# --- Functions ---

def generate_text_gemini(prompt):
    """
    Connects to Gemini 2.0 Flash (2026 Standard).
    """
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    # UPDATED: Using gemini-2.0-flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
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
            # Fallback: If 2.0 isn't live yet in your region, try 'gemini-pro' (The alias)
            print(f"2.0 Flash failed, trying gemini-pro alias...")
            url_backup = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
            response_backup = requests.post(url_backup, headers=headers, json=payload)
            
            if response_backup.status_code == 200:
                return response_backup.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                return f"Gemini Error ({response.status_code}): {response.text}"
                
    except Exception as e:
        return f"Connection Error: {e}"

def generate_image_hf(prompt):
    """
    Uses Stable Diffusion v1.5 (RunwayML).
    This is the most reliable free model.
    """
    try:
        image = hf_client.text_to_image(
            f"cinematic movie poster for {prompt}, high quality",
            model="runwayml/stable-diffusion-v1-5"
        )
        return image, "SD v1.5"
    except Exception as e:
        # We return the specific error 'e' to the UI so we can see it
        return None, str(e)

def generate_video_hf(prompt):
    """Experimental Video Generation."""
    try:
        video_bytes = hf_client.text_to_video(
            prompt,
            model="damo-vilab/text-to-video-ms-1.7b"
        )
        return video_bytes
    except Exception:
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="AI Studio 2026", page_icon="🚀")

st.title("🚀 AI Studio 2026")
st.markdown("Text via **Gemini 2.0** | Images via **SD v1.5**")

user_prompt = st.text_input("Enter content idea:", placeholder="e.g., A robot painting in the rain")

if st.button("Generate Content"):
    if not user_prompt:
        st.warning("Please enter a prompt!")
    else:
        col1, col2 = st.columns(2)

        # 1. TEXT (Gemini 2.0)
        with st.spinner("Gemini 2.0 is thinking..."):
            caption = generate_text_gemini(user_prompt)
            if "Error" in caption:
                st.error("Text Generation Failed")
                st.code(caption) # Shows the full error details
            else:
                st.success("✅ Caption Ready")
                st.markdown(f"### 📢 {caption}")

        # 2. IMAGE (SD v1.5)
        with st.spinner("Generating Poster..."):
            img, error_msg = generate_image_hf(user_prompt)
            if img:
                with col1:
                    st.image(img, caption=f"Poster ({error_msg})", use_container_width=True)
                st.success("✅ Poster Ready")
            else:
                with col1:
                    st.error("Image Failed. Details:")
                    st.warning(error_msg) # Prints the exact reason

        # 3. VIDEO
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
                    st.warning("Video Server Busy.")
