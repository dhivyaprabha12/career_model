import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from PyPDF2 import PdfReader
import time

st.set_page_config(page_title="Career AI", layout="wide")

MODEL_PATH = "models/career_support_model"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


st.markdown("""
<style>

.stApp {
    background-color: #000000;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
    font-size: 18px;
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    background-color: #000000;
    border-right: 1px solid #222222;
    color: #ffffff;
    font-weight: 700;
}

.stChatMessage {
    background: #000000;
    border: 1px solid #222222;
    border-radius: 14px;
    padding: 16px;
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
}

.stChatInput textarea {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #333333;
    font-size: 18px;
    font-weight: 700;
}

p, span, div, label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.to("cpu")
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# ------------------ SESSION ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ PDF ------------------
def extract_text(file):
    reader = PdfReader(file)
    return " ".join([p.extract_text() or "" for p in reader.pages]).strip()

# ------------------ 🔥 STRONG SYSTEM PROMPT (FIXED) ------------------
def build_prompt(user_input, history, category):

    system_prompt = f"""
You are an expert career advisor AI.

STRICT RULES:
- Respond ONLY to the user's latest question
- Do NOT generate conversations (no User:, Former:, Assistant:)
- Do NOT repeat training-style dialogue
- Give clear, structured, professional answers only
- Be concise and practical

OUTPUT FORMAT:
1. Career Direction (short and clear)
2. Required Skills (bullet points)
3. Tools / Technologies
4. Step-by-step Roadmap
5. Final Advice

Category: {category}
"""

    convo = ""
    for h in history:
        convo += f"User: {h['user']}\nAssistant: {h['bot']}\n"

    return f"{system_prompt}\n\n{convo}User: {user_input}\nAssistant:"

# ------------------ GENERATE (FIXED CONTROL) ------------------
def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.3,
            top_p=0.8,
            repetition_penalty=1.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    text = text.split("Assistant:")[-1].strip()

    # REMOVE BAD PATTERNS
    for w in ["User:", "Former:", "Assistant:"]:
        text = text.replace(w, "")

    return text.strip()

# ------------------ STREAM ------------------
def stream(text):
    placeholder = st.empty()
    out = ""
    for w in text.split():
        out += w + " "
        placeholder.markdown(out)
        time.sleep(0.01)

# ------------------ SIDEBAR ------------------
st.sidebar.title("Career AI")

category = st.sidebar.radio(
    "Focus Area",
    ["General", "Skills", "Jobs", "Higher Studies", "Interview"]
)

uploaded_file = st.sidebar.file_uploader("Upload Resume (PDF)")

# ------------------ HEADER ------------------
st.title("Career AI")
st.caption("AI-powered career guidance assistant")

# ------------------ CHAT HISTORY ------------------
for chat in st.session_state.history:
    with st.chat_message("user"):
        st.write(chat["user"])
    with st.chat_message("assistant"):
        st.write(chat["bot"])

# ------------------ INPUT ------------------
user_input = st.chat_input("Ask your question...")

# ------------------ HANDLE RESUME ------------------
resume_mode = False
resume_text = ""

if uploaded_file:
    resume_text = extract_text(uploaded_file)
    if resume_text:
        resume_mode = True
        user_input = "Analyze my resume and give career roadmap"

# ------------------ RESPONSE ------------------
if user_input:
    with st.chat_message("user"):
        st.write("Resume uploaded" if resume_mode else user_input)

    final_input = user_input if not resume_mode else f"""
Resume:
{resume_text}
"""

    prompt = build_prompt(final_input, st.session_state.history, category)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate(prompt)
            stream(response)

    st.session_state.history.append({
        "user": "Resume Analysis" if resume_mode else user_input,
        "bot": response
    })