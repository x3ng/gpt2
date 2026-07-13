from transformers import pipeline
import os
from config import BASE_DIR

from transformers.utils import logging
logging.set_verbosity_error()

MODEL_PATH = os.path.join(BASE_DIR, "result", "gpt2-117m-final")

print(f"📂 Loading model from {MODEL_PATH}...")

generator = pipeline(
    'text-generation',
    model=MODEL_PATH,
    device_map="auto"
)

print("✅ Model loaded successfully")
print("\n===== Chat Started (type 'exit' to quit) =====\n")

while True:
    user_input = input("👤 You: ").strip()
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("👋 Goodbye!")
        break
    
    if not user_input:
        continue

    result = generator(
        user_input,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.15,
    )

    response = result[0]['generated_text'][len(user_input):].strip()
    print(f"🤖 Assistant: {response}\n")
