import streamlit as st
import google.generativeai as genai
import replicate
import os

# --- Configuration ---
#Ideally, store these in a .env file or Streamlit secrets
os.environ["REPLICATE_API_TOKEN"] = "YOUR_REPLICATE_API_KEY"
GOOGLE_API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY"

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# --- Functions ---

def generate_catchy_phrase(prompt):
    """Generates a short, catchy social media caption using Gemini."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        f"Write a single, short, punchy, and viral social media caption (under 15 words) for a video about: {prompt}. No hashtags, just the phrase."
    )
    return response.text.strip()

def generate_poster(prompt):
    """Generates a poster using Flux-Schnell (Great for text/composition)."""
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": f"A high quality movie poster style image of {prompt}, cinematic lighting, 8k, vibrant colors, typography",
            "aspect_ratio": "9:16", # Vertical for TikTok/Reels
            "output_format": "webp",
            "output_quality": 90
        }
    )
    # Replicate returns a list of URLs
    return output[0]

def generate_video(prompt):
    """Generates a video using CogVideoX-5B (State of the art open model)."""
    output = replicate.run(
        "thudm/cogvideox-5b",
        input={
            "prompt": f"Cinematic motion, high quality, 4k: {prompt}",
            "num_frames": 49,
            "guidance_scale": 6,
            "num_inference_steps": 50
        }
    )
    return output # Usually a URL to an mp4 file

# --- Streamlit UI ---

st.set_page_config(page_title="AI Content Machine", page_icon="🎬")

st.title("🎬 AI Content Generator")
st.markdown("Turn a single idea into a **Poster**, a **Phrase**, and a **Video**.")

# User Input
user_prompt = st.text_input("Enter your content idea:", placeholder="e.g., A futuristic astronaut discovering a glowing flower on Mars")

if st.button("Generate Content"):
    if not user_prompt:
        st.warning("Please enter a prompt first!")
    else:
        st.info("🚀 Starting the creative engines...")
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])

        # 1. Generate Phrase (Fastest)
        with st.spinner("Writing catchy phrase..."):
            try:
                catchy_phrase = generate_catchy_phrase(user_prompt)
                st.success("Phrase Generated!")
                st.markdown(f"### 📢 {catchy_phrase}")
            except Exception as e:
                st.error(f"Error generating text: {e}")

        # 2. Generate Poster
        with st.spinner("Designing poster..."):
            try:
                poster_url = generate_poster(user_prompt)
                with col1:
                    st.image(poster_url, caption="Generated Poster", use_container_width=True)
            except Exception as e:
                st.error(f"Error generating poster: {e}")

        # 3. Generate Video (Slowest)
        with st.spinner("Rendering video (this takes a moment)..."):
            try:
                video_url = generate_video(user_prompt)
                with col2:
                    st.video(video_url)
            except Exception as e:
                st.error(f"Error generating video: {e}")
                
        st.balloons()
