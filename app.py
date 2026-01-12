import streamlit as st
from huggingface_hub import InferenceClient
import tempfile

# --- Configuration ---
if "HF_TOKEN" in st.secrets:
    client = InferenceClient(token=st.secrets["HF_TOKEN"])
else:
    st.error("Missing HF_TOKEN in secrets!")
    st.stop()

# --- PIPELINE 1: TEXT (Lightweight & Fast) ---
def pipeline_text(prompt):
    """
    Uses Google's Flan-T5-Base.
    This model is small (250MB) vs others (10GB+), 
    so it loads instantly and rarely times out on free tier.
    """
    try:
        # T5 is an encoder-decoder model, perfect for "instructions"
        input_text = f"write a viral caption about {prompt}"
        
        response = client.text_generation(
            input_text,
            model="google/flan-t5-base", 
            max_new_tokens=50,
            temperature=0.7
        )
        return response
        
    except Exception as e:
        # Return the exact error so we can see it in the UI
        return f"Error: {e}"

# --- PIPELINE 2: IMAGE (Flux with Fallback) ---
def pipeline_image(prompt):
    try:
        image = client.text_to_image(
            f"Movie poster for {prompt}, cinematic, 8k, highly detailed",
            model="black-forest-labs/FLUX.1-dev"
        )
        return image, "Flux"
    except Exception:
        # Fallback to SDXL if Flux is busy
        image = client.text_to_image(
            f"Movie poster for {prompt}",
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )
        return image, "SDXL"

# --- PIPELINE 3: VIDEO ---
def pipeline_video(prompt):
    return client.text_to_video(
        prompt,
        model="damo-vilab/text-to-video-ms-1.7b"
    )

# --- Streamlit UI ---

st.set_page_config(page_title="AI Pipeline Studio", page_icon="⚡")

st.title("⚡ The 3-Pipeline Studio")
st.markdown("Using **Fast Models** (Flan-T5) to prevent timeouts.")

user_prompt = st.text_input("Enter content idea:", placeholder="e.g., A cybernetic tiger running in neon rain")

if st.button("Run Pipelines"):
    if not user_prompt:
        st.warning("Input required!")
    else:
        col1, col2 = st.columns(2)
        
        # 1. RUN TEXT PIPELINE
        with st.spinner("Pipeline 1: Text..."):
            caption = pipeline_text(user_prompt)
            
            # Check for error string
            if caption.startswith("Error:"):
                st.error("Text Generation Failed. Details:")
                st.code(caption) # This will print the EXACT error message
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
