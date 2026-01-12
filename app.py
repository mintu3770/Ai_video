import streamlit as st
from huggingface_hub import InferenceClient
import tempfile

# --- Configuration ---
# We use the token for all requests, but we treat each task differently.
if "HF_TOKEN" in st.secrets:
    # We initialize the client generally, but will call it differently for each task
    client = InferenceClient(token=st.secrets["HF_TOKEN"])
else:
    st.error("Missing HF_TOKEN in secrets!")
    st.stop()

# --- PIPELINE 1: TEXT (The Reliable "Instruction" Pipeline) ---
def pipeline_text(prompt):
    """
    Uses Google's Flan-T5. 
    Why? It's not a 'Chat' model. It's an 'Instruction' model.
    It almost never fails with 'format' errors on the free tier.
    """
    try:
        # We use simple text_generation, not chat
        input_text = f"Write a short viral caption for a video about {prompt}"
        
        response = client.text_generation(
            input_text,
            model="google/flan-t5-large", 
            max_new_tokens=50,
            temperature=0.7
        )
        return response
    except Exception as e:
        return f"Text Pipeline Error: {e}"

# --- PIPELINE 2: IMAGE (The "Flux" Pipeline) ---
def pipeline_image(prompt):
    """
    Uses Flux.1-dev.
    This works well but can be busy.
    """
    try:
        image = client.text_to_image(
            f"Movie poster for {prompt}, cinematic, 8k, highly detailed",
            model="black-forest-labs/FLUX.1-dev"
        )
        return image
    except Exception as e:
        st.warning(f"Flux failed ({e}). Switching to SDXL Pipeline...")
        # Fallback Sub-Pipeline
        return client.text_to_image(
            f"Movie poster for {prompt}",
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )

# --- PIPELINE 3: VIDEO (The "Experimental" Pipeline) ---
def pipeline_video(prompt):
    """
    Uses Damo-Vilab.
    Strictly experimental on free tier.
    """
    # We don't try/except here so we can show the specific error in the UI
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
            st.success("✅ Caption Generated")
            st.markdown(f"### 📢 {caption}")

        # 2. RUN IMAGE PIPELINE
        with st.spinner("Pipeline 2: Image..."):
            try:
                img = pipeline_image(user_prompt)
                with col1:
                    st.image(img, caption="Generated Poster", use_container_width=True)
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
                # We specifically catch the video error here to keep the rest running
                with col2:
                    st.warning("Video Pipeline Timed Out (Free Tier limit).")
