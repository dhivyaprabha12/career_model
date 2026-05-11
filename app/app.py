from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# ✅ Correct paths
model_path = "./career_support_model"
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ✅ Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

# ✅ Attach LoRA adapter
model = PeftModel.from_pretrained(base_model, model_path)

device = "cpu"
model.to(device)

# ✅ Chat memory
chat_history = []

# ✅ Build prompt
def build_prompt(user_input, history):
    system_prompt = (
        "You are a professional AI Career Assistant. "
        "Provide structured, practical, and helpful advice."
    )

    conversation = ""
    for h in history:
        conversation += f"User: {h['user']}\nAssistant: {h['bot']}\n"

    return f"{system_prompt}\n\n{conversation}User: {user_input}\nAssistant:"

# ✅ Clean response
def clean_response(text):
    return text.split("\nUser:")[0].strip()

# ✅ Generate response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=1.2,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_response(response.split("Assistant:")[-1])

# ✅ Chat loop
while True:
    user_input = input("\n👤 You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    prompt = build_prompt(user_input, chat_history)

    bot_response = generate_response(prompt)

    print("🤖 Bot:", bot_response)

    chat_history.append({"user": user_input, "bot": bot_response})

    # Limit memory
    if len(chat_history) > 5:
        chat_history.pop(0) 