import streamlit as st
from huggingface_hub import InferenceClient
import tempfile

# --- Configuration & Secrets ---
# Ensure "HF_TOKEN" is set in your .streamlit/secrets.toml file.
# We no longer need the Google API key.
if "HF_TOKEN" in st.secrets:
    hf_client = InferenceClient(token=st.secrets["HF_TOKEN"])
else:
    st.error("Missing HF_TOKEN. Please add it to your secrets.")
    st.stop()

# --- Functions ---

def generate_catchy_phrase(prompt):
    """Generates text using Hugging Face Chat Completion (Mistral v0.3)."""
    try:
        # Define the message structure for the Chat Model
        messages = [
            {"role": "user", "content": f"Write a single, short, punchy, viral social media caption (under 15 words) for a video about: {prompt}. No hashtags, just the phrase."}
        ]
        
        # Use chat_completion (Required for Instruct models)
        response = hf_client.chat_completion(
            messages,
            model="mistralai/Mistral-7B-Instruct-v0.3",
            max_tokens=60,
            temperature=0.7
        )
        
        # Extract the content from the response object
        return response.choices[0].message.content.strip('"')

    except Exception as e:
        return f"Text Error: {e}"

def generate_poster(prompt):
    """Generates an image using Flux.1-dev with fallback to SDXL."""
    try:
        # Attempt 1: Flux (High Quality)
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
    """Generates a video using Damo-Vilab."""
    # Note: This often times out on the free tier (503 error)
    video_bytes = hf_client.text_to_video(
        prompt,
        model="damo-vilab/text-to-video-ms-1.7b"
    )
    return video_bytes

# --- Streamlit UI ---

st.set_page_config(page_title="Free AI Content Tool", page_icon="🎓")

st.title("🎓 The Student's AI Studio")
st.markdown("Generate content for **$0** using Hugging Face (Text, Image & Video).")

user_prompt = st.text_input("Enter your content idea:", placeholder="e.g., A robot painting a canvas in space")

if st.button("Generate for Free"):
    if not user_prompt:
        st.warning("Please enter a prompt first!")
    else:
        st.info("🚀 Creating content...")
        
        col1, col2 = st.columns([1, 1])

        # 1. Text Generation (Mistral Chat)
        with st.spinner("Writing Caption..."):
            phrase = generate_catchy_phrase(user_prompt)
            st.success("Caption Ready!")
            st.markdown(f"### 📢 {phrase}")

        # 2. Image Generation (Flux or SDXL)
        with st.spinner("Generating Poster..."):
            try:
                poster_image, model_used = generate_poster(user_prompt)
                with col1:
                    st.image(poster_image, caption=f"Poster ({model_used})", use_container_width=True)
            except Exception as e:
                with col1:
                    st.error(f"Image generation failed: {e}")

        # 3. Video Generation (Experimental)
        with st.spinner("Generating Video (May timeout on free tier)..."):
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
                    st.warning(f"Video skipped (Free tier limit). Details: {e}")
