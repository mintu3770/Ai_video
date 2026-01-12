import streamlit as st
import requests
import urllib.parse
import random
from huggingface_hub import InferenceClient
import tempfile

# --- Configuration ---
# Video generation requires a Hugging Face Token.
# If it's missing, we just disable the video part instead of crashing.
hf_client = None
if "HF_TOKEN" in st.secrets:
    hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])

# --- Functions ---

def generate_text_pollinations(prompt):
    """Generates text via Pollinations (Unlimited)."""
    try:
        clean_prompt = urllib.parse.quote(f"Write a short, viral caption for: {prompt}")
        url = f"https://text.pollinations.ai/{clean_prompt}"
        response = requests.get(url)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def get_image_url(prompt):
    """Generates an Image URL via Pollinations (Unlimited)."""
    encoded_prompt = urllib.parse.quote(prompt)
    seed = random.randint(1, 99999)
    # Using 'flux' model for best quality
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&seed={seed}&model=flux&nologo=true"

def download_image(url):
    """Downloads image bytes for the save button."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        return None
    except:
        return None

def generate_video_hf(prompt):
    """Generates a video via Hugging Face (Requires Token)."""
    if not hf_client:
        return None, "Missing HF_TOKEN in secrets."
    
    try:
        # Using the standard free video model
        video_bytes = hf_client.text_to_video(
            prompt,
            model="damo-vilab/text-to-video-ms-1.7b"
        )
        return video_bytes, None
    except Exception as e:
        return None, str(e)

# --- Streamlit UI ---

st.set_page_config(page_title="Ultimate AI Studio", page_icon="🎬", layout="wide")

st.title("🎬 Ultimate AI Studio")
st.markdown("Text & Image (Unlimited) | Video (Hugging Face)")

user_prompt = st.text_input("Enter content idea:", placeholder="e.g., A cyberpunk samurai eating ramen")

if st.button("Generate Everything"):
    if not user_prompt:
        st.warning("Please enter a prompt!")
    else:
        st.info("🚀 Generating Content...")
        
        # Create 3 columns for a full dashboard view
        col1, col2, col3 = st.columns(3)

        # 1. TEXT
        with col1:
            st.subheader("📝 Text")
            with st.spinner("Writing..."):
                caption = generate_text_pollinations(user_prompt)
                st.success("Caption Ready")
                st.info(caption)

        # 2. IMAGE
        with col2:
            st.subheader("🖼️ Image")
            with st.spinner("Drawing..."):
                img_url = get_image_url(user_prompt)
                st.image(img_url, caption="Pollinations Flux", use_container_width=True)
                
                # Download Button
                img_data = download_image(img_url)
                if img_data:
                    st.download_button("⬇️ Save Image", img_data, "poster.jpg", "image/jpeg")

        # 3. VIDEO
        with col3:
            st.subheader("🎥 Video")
            if not hf_client:
                st.warning("⚠️ Video disabled. Add `HF_TOKEN` to secrets to enable.")
            else:
                with st.spinner("Rendering (May take 30s)..."):
                    vid_bytes, error = generate_video_hf(user_prompt)
                    
                    if vid_bytes:
                        # Save to temp file to display
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                            tfile.write(vid_bytes)
                            vid_path = tfile.name
                        st.video(vid_path)
                        st.success("Video Ready")
                    else:
                        st.error("Video Failed")
                        st.warning(f"Error: {error}")
                        st.caption("Note: Free video servers often timeout. Try again in 1 minute.")
