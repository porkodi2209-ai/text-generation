import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Text Generation App",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .output-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #444;
        font-size: 18px;
        line-height: 1.7;
    }

    .about-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #444;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="main-title">🤖 AI Text Generation App</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Generate text using Python, Streamlit and a Hugging Face language model.'
    '</div>',
    unsafe_allow_html=True
)


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

@st.cache_resource
def load_model():
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1
    )

    return generator

with st.spinner("Loading AI model... Please wait"):
    generator = load_model()


st.markdown(
    '<div class="section-title">Enter your prompt:</div>',
    unsafe_allow_html=True
)

prompt = st.text_area(
    "",
    placeholder="Example: Explain Artificial Intelligence in simple English.",
    height=150
)


col1, col2 = st.columns(2)

with col1:

    max_tokens = st.slider(
        "Maximum new tokens:",
        min_value=50,
        max_value=300,
        value=150,
        step=10
    )

with col2:

    temperature = st.slider(
        "Temperature:",
        min_value=0.10,
        max_value=1.00,
        value=0.70,
        step=0.05
    )


if st.button("✨ Generate Text", use_container_width=True):

    if prompt.strip() == "":
        st.warning("Please enter a prompt first.")

    else:

        with st.spinner("Generating text..."):

            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            formatted_prompt = generator.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            result = generator(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.90,
                repetition_penalty=1.20,
                no_repeat_ngram_size=3,
                return_full_text=False
            )

            generated_text = result[0]["generated_text"].strip()

        

        st.markdown(
            '<div class="section-title">✨ Generated Text</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="output-box">{generated_text}</div>',
            unsafe_allow_html=True
        )



st.divider()
st.markdown(
    '<div class="section-title">📌 About this App</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="about-box">

    <p>
    This application uses <b>Python, Streamlit and Hugging Face Transformers</b>
    for AI-based text generation.
    </p>

    <p>
    The language model is downloaded and executed locally on the computer.
    </p>

    <p>
    <b>Hugging Face API token is not required.</b>
    </p>

    <p>
    The application allows users to enter a prompt and generate text by
    controlling the maximum number of tokens and temperature.
    </p>

    <p>
    <b>Model:</b> SmolLM2-360M-Instruct
    </p>

    </div>
    """,
    unsafe_allow_html=True
)