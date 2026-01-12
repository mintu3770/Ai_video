import streamlit as st
from huggingface_hub import InferenceClient
import tempfile

# --- Configuration ---
if "HF_TOKEN" in st.secrets:
    client = InferenceClient(token=st.secrets["HF_TOKEN"])
else:
    st.error("Missing HF_TOKEN in secrets!")
    st.stop()

# --- PIPELINE 1: TEXT (Dual-Model Fallback) ---
def pipeline_text(prompt):
    """
    Tries Microsoft Phi-2 (Smart). 
    If that fails, falls back to GPT-2 (Reliable).
    """
    # Attempt 1: Microsoft Phi-2 (Smart & Fast)
    try:
        # Phi-2 works best with a "completion" style prompt
        input_text = f"Instruct: Write a short, viral social media caption for a video about {prompt}.\nOutput:"
        
        response = client.text_generation(
            input_text,
            model="microsoft/phi-2", 
            max_new_tokens=50,
            temperature=0.7,
            stop_sequences=["\n"] # Stop it from rambling
        )
        return response.strip()
        
    except Exception as e1:
        # Attempt 2: GPT-2 (The "Old Reliable" fallback)
        try:
            print(f"Phi-2 failed ({e1}). Switching to GPT-2...")
            response = client.text_generation(
                f"Video caption for {prompt}:", # Simple prompt for GPT-2
                model="gpt2", 
                max_new_tokens=30
            )
            return response.strip()
        except Exception as e2:
            return f"Error: Both models failed. Details: {e2}"

# --- PIPELINE 2: IMAGE (Flux) ---
def pipeline_image(prompt):
    """Uses Flux.1-dev with SDXL fallback."""
    try:
        image = client.text_to_image(
            f"Movie poster for {prompt}, cinematic, 8k, highly detailed",
            model="black-forest-labs/FLUX.1-dev"
        )
        return image, "Flux"
    except Exception:
        # Fallback to SDXL
        image = client.text_to_image(
            f"Movie poster for {prompt}",
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )
        return image, "SDXL"

# --- PIPELINE 3: VIDEO ---
def pipeline_video(prompt):
    """Experimental Video Generation."""
    return client.text_to_video(
        prompt,
        model="damo-vilab/text-to-video-ms-1.7b"
    )

# --- Streamlit UI ---

st.set_page_config(page_title="AI Pipeline Studio", page_icon="⚡")

st.title("⚡ The 3-Pipeline Studio")
st.markdown("Using specialized models for Text, Image, and Video.")

user_prompt = st.text_input("Enter content idea:", placeholder="e.g., A cybernetic tiger running in neon rain")

if st.button("Run Pipelines"):
    if not user_prompt:
        st.warning("Input required!")
    else:
        col1, col2 = st.columns(2)
        
        # 1. RUN TEXT PIPELINE
        with st.spinner("Pipeline 1: Text..."):
            caption = pipeline_text(user_prompt)
            
            # Check if the result is an error message
            if "Error:" in caption:
                st.error("Text Pipeline Failed")
                st.write(caption) # Show the error details
            else:
                st.success("✅ Caption Generated")
                st.markdown(f"### 📢 {caption}")

        # 2. RUN IMAGE PIPELINE
        with st.spinner("Pipeline 2: Image..."):
            try:
                img, model_name = pipeline_image(user_prompt)
                with col1:
                    st.image(img, caption=f"Poster ({model_name})", use_container_width=True)
                st.success("✅ Image Generated")
            except Exception as e:
                st.error(f"Image Pipeline Failed: {e}")

        # 3. RUN VIDEO PIPELINE
        with st.spinner("Pipeline 3: Video (Heavy)..."):
            try:
                vid_bytes = pipeline_video(user_prompt)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(vid_bytes)
                    vid_path = tfile.name
                with col2:
                    st.video(vid_path)
                st.success("✅ Video Generated")
            except Exception as e:
                with col2:
                    st.warning("Video Pipeline Timed Out (Free Tier limit).")
