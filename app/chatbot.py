import operator

from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import List, TypedDict
# from googlesearch import search

load_dotenv()
client = OpenAI()

# Step 1: GPU metadata (inline for now)
with open('./index/hardware_metadata.json', 'r') as f:
    gpu_data = json.load(f)
# gpu_data = [
#     {
#         "filename": "nvidia_4060.jpg",
#         "path": "./data/hardware_images/nvidia_4060.jpg",
#         "brand": "NVIDIA",
#         "model": "RTX 4060",
#         "memory": "8GB",
#         "known_issues": [
#             "Drivers may crash in certain games",
#             "High temperatures under load"
#         ],
#         "solutions": [
#             "Update to the latest NVIDIA driver from official website",
#             "Ensure proper cooling and airflow in the case",
#             "Reduce in-game graphics settings if overheating persists"
#         ]
#     },
#     {
#         "filename": "nvidia_4070.jpg",
#         "path": "./data/hardware_images/nvidia_4070.jpg",
#         "brand": "NVIDIA",
#         "model": "RTX 4070",
#         "memory": "12GB",
#         "known_issues": [
#             "Occasional screen flickering",
#             "Compatibility issues with older motherboards"
#         ],
#         "solutions": [
#             "Update BIOS to the latest version",
#             "Use certified HDMI/DisplayPort cables",
#             "Update drivers and check for Windows updates"
#         ]
#     }
# ]

# Step 2: User query
user_query = "Tell me about the NVIDIA RTX 4060"

# Step 3: Search JSON for the GPU
selected_gpu = None
for gpu in gpu_data:
    if gpu["model"].lower() in user_query.lower():
        selected_gpu = gpu
        break

# Step 4: If found, pass metadata into context
if selected_gpu:
    context_text = f"""
    GPU: {selected_gpu['brand']} {selected_gpu['model']}
    Memory: {selected_gpu['memory']}

    Known Issues:
    - {"; ".join(selected_gpu['known_issues'])}

    Solutions:
    - {"; ".join(selected_gpu['solutions'])}
    """
else:
    context_text = "No matching GPU found in database."

# Step 5: Send to LLM
def llm_run(query):
    return client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful GPU customer support assistant."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": context_text}
        ],
        max_tokens=500,
    )

class Message(TypedDict):
    role: str
    content: str

def run_loop():
    past_messages: List[Message] = []
    try:
        query = input("Enter your query regarding the product you searched: ")
        while query != "exit":
            past_messages.append({"role": "user", "content": query})
            response = llm_run(past_messages).choices[0].message.content
            print(f"AI: {response}")
            past_messages.append({"role": "assistant", "content": response})
            query = input("\nEnter your next query or type 'exit' to quit: ")

        if query == "exit":
            print("\nThank you for using the GPU support assistant. Hope we could solve your issue. Goodbye!\n")
            return

    except KeyboardInterrupt:
        print("\nExiting...")
        print("\nThank you for using the GPU support assistant. Hope we could solve your issue. Goodbye!\n")
        return


# print(response.choices[0].message.content)

if __name__ == "__main__":
    # query = "NVIDIA RTX 4060 known issues"
    #
    # # returns top 5 search results
    # result = list(search(query))
    # print(result)
    run_loop()
