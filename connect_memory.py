from huggingface_hub import InferenceClient

# Your Hugging Face token (keep this secure!)
HF_TOKEN = "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"

# Initialize the InferenceClient with your token
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=HF_TOKEN
)

# System prompt with safety rule
system_prompt = (
    "You are MedBot, a helpful and cautious medical assistant. "
    "Only answer questions if you're confident based on medical evidence. "
    "If you're not sure, respond with 'I'm not sure. Please consult a doctor.'"
)

# Chat history and new user input
chat_history = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What are the symptoms of diabetes?"}
]

# Send the request using .chat_completion()
response = client.chat_completion(
    messages=chat_history,
    max_tokens=300,
    temperature=0.5,
    top_p=0.9,
    repetition_penalty=1.1
)

# Print the model's response
print("MedBot:", response.choices[0].message["content"])
