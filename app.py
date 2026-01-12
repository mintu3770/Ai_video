import streamlit as st
from huggingface_hub import InferenceClient
import time
import urllib.parse

# --- Configuration ---
if "HF_TOKEN" in st.secrets:
    hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
else:
    st.error("Missing HF_TOKEN. Please add it to your secrets!")
    st.stop()

# --- Functions ---

def generate_text_hf(prompt):
    """
    Uses Google's Flan-T5-Large via Hugging Face.
    This bypasses the Gemini 429/404 errors completely.
    """
    try:
        # Flan-T5 is an 'Instruction' model, not a 'Chat' model.
        # It works reliably with the basic text_generation API.
        input_text = f"Write a viral caption for a video about {prompt}"
        
        response = hf_client.text_generation(
            input_text,
            model="google/flan-t5-large",
            max_new_tokens=50,
            temperature=0.7
        )
        return response
    except Exception as e:
        return f"Text Error: {e}"

def generate_image_pollinations(prompt):
    """
    Generates an image via Pollinations.ai (Unlimited).
    Returns both the URL and a 'Clickable' version in case the image fails to load.
    """
    # 1. URL Encode the prompt to prevent breaking the link
    clean_prompt = urllib.parse.quote(prompt)
    
    # 2. Add random seed to force new image generation
    seed = int(time.time())
    
    # 3. Construct URL
    url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width=1024&height=1536&seed={seed}&nologo=true"
    return url

def generate_video_hf(prompt):
    """
    Video generation (Bonus).
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

st.set_page_config(page_title="Fail-Safe AI Studio", page_icon="🛡️")

st.title("🛡️ Fail-Safe AI Studio")
st.markdown("Bypassing Google Limits with **Flan-T5** & **Pollinations**")

user_prompt = st.text_input("Enter content idea:", placeholder="e.g., A cyberpunk samurai in neon rain")

if st.button("Generate Content"):
    if not user_prompt:
        st.warning("Please enter a prompt!")
    else:
        st.info("🚀 Generating...")
        
        col1, col2 = st.columns(2)

        # 1. TEXT (Hugging Face / Flan-T5)
        with st.spinner("Writing Caption..."):
            caption = generate_text_hf(user_prompt)
            if "Error" in caption:
                st.error(caption)
            else:
                st.success("✅ Caption Ready")
                st.markdown(f"### 📢 {caption}")

        # 2. IMAGE (Pollinations)
        with st.spinner("Generating Image..."):
            img_url = generate_image_pollinations(user_prompt)
            
            with col1:
                # We try to show the image
                st.image(img_url, caption="Pollinations Image", use_container_width=True)
                # Backup Link: If the image above is broken, the user can click this
                st.markdown(f"[**🔗 Click here if image doesn't load**]({img_url})")
            
            st.success("✅ Image Link Ready")

        # 3. VIDEO (Hugging Face)
        with st.spinner("Rendering Video (May be busy)..."):
            vid_bytes = generate_video_hf(user_prompt)
            if vid_bytes:
                # Write to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(vid_bytes)
                    vid_path = tfile.name
                
                with col2:
                    st.video(vid_path)
                st.success("✅ Video Ready")
            else:
                with col2:
                    st.warning("Video Server Busy (Free Tier).")
