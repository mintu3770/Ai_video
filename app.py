import streamlit as st
import replicate
import requests
import tempfile
import os

# --- Configuration & Secrets ---
# 1. Google (Text)
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing GOOGLE_API_KEY in secrets.")
    st.stop()

# 2. Replicate (Images & Video)
if "REPLICATE_API_TOKEN" not in st.secrets:
    st.error("Missing REPLICATE_API_TOKEN in secrets.")
    st.stop()

# Set the environment variable for the Replicate library
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

# --- Functions ---

def generate_gemini_text(prompt):
    """
    Connects to Google's API using Gemini 2.5 Flash.
    """
    api_key = st.secrets["GOOGLE_API_KEY"]
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": f"Write a single, short, punchy, viral social media caption (under 15 words) for a video about: {prompt}. No hashtags, just the phrase."}]
        }]
    }

    # Using the standard 2026 model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            return f"Gemini Error: {response.text}"
    except Exception as e:
        return f"Connection Error: {e}"

def generate_replicate_image(prompt):
    """
    Generates an image using Flux Schnell on Replicate.
    """
    try:
        # Flux Schnell is super fast and costs roughly $0.003 per image
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": f"cinematic movie poster for {prompt}, high quality, typography, 8k",
                "go_fast": True,  # Optimizes for speed
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "2:3", # Poster ratio
                "output_format": "webp",
                "output_quality": 80
            }
        )
        # Replicate returns a list of output URLs/Streams
        return output[0], "Flux Schnell"
    except Exception as e:
        return None, f"Replicate Error: {e}"

def generate_replicate_video(prompt):
    """
    Generates a video using Luma Ray or similar on Replicate.
    """
    try:
        # Using a standard fast video model (ZeroScope is cheaper/faster than others)
        output = replicate.run(
            "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351",
            input={
                "prompt": prompt,
                "num_frames": 24,
                "width": 576,
                "height": 320
            }
        )
        return output[0] # Returns the video URL
    except Exception as e:
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="Replicate + Gemini Studio", page_icon="⚡")

st.title("⚡ The High-Speed Studio")
st.markdown("Text by **Gemini 2.5** | Visuals by **Replicate (Flux)**")

user_prompt = st.text_input("Enter content idea:", placeholder="e.g., A cyberpunk detective in rainy tokyo")

if st.button("Generate Content"):
    if not user_prompt:
        st.warning("Please enter a prompt first!")
    else:
        st.info("🚀 Generating content on high-speed servers...")
        
        col1, col2 = st.columns(2)

        # 1. TEXT (Gemini)
        with st.spinner("Writing Caption..."):
            caption = generate_gemini_text(user_prompt)
            if "Error" in caption:
                st.error("Text Generation Failed")
                st.write(caption)
            else:
                st.success("✅ Caption Ready")
                st.markdown(f"### 📢 {caption}")

        # 2. IMAGE (Replicate Flux)
        with st.spinner("Generating Poster (Flux)..."):
            img_url, model_name = generate_replicate_image(user_prompt)
            if img_url:
                with col1:
                    st.image(img_url, caption=f"Poster ({model_name})", use_container_width=True)
                st.success("✅ Poster Ready")
            else:
                with col1:
                    st.error(f"Image Failed: {model_name}")

        # 3. VIDEO (Replicate ZeroScope)
        with st.spinner("Rendering Video..."):
            vid_url = generate_replicate_video(user_prompt)
            if vid_url:
                with col2:
                    st.video(vid_url)
                st.success("✅ Video Ready")
            else:
                with col2:
                    st.warning("Video failed (Check Replicate credits).")
