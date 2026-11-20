import ollama

prompt = """
Classify the skin condition present in the image. The skin condition is one of the three
labels: Eczema, Insect Bite, or Folliculitis. Respond with exactly one label.
"""

response = ollama.chat(
    model="gemma4:e4b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)
print("Prompt tokens:", response.get("prompt_eval_count"))
print("Response:")
print(response)