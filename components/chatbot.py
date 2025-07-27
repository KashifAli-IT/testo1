import gradio as gr
from groq import Groq
import os

# ✅ Set your Groq API key (store as secret on Hugging Face Spaces if possible)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
client = Groq(api_key=GROQ_API_KEY)

# ✅ Chat function with OpenAI-style memory and English-only logic
def chat_with_memory_english(user_input, history):
    system_prompt = (
        "You are FinGenius, a friendly financial planning assistant for young people in Pakistan. "
        "Always respond in English only. Provide budgeting advice, saving tips, and basic investment suggestions. "
        "Avoid using any non-English language or mixed-language replies."
    )

    if history is None:
        history = []

    messages = [{"role": "system", "content": system_prompt}] + history
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        bot_reply = response.choices[0].message.content.strip()
    except Exception as e:
        bot_reply = f"❌ Error: {str(e)}"

    # Update memory
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": bot_reply})

    return history, history, ""  # 🔁 Clear input textbox automatically

with gr.Blocks(css="""
body {
    background-color: #f3f3f3;
    font-family: 'Segoe UI', sans-serif;
}
.container {
    max-width: 700px;
    margin: 0 auto;
    padding-top: 40px;
}
gr-chatbot {
    border: 1px solid #ddd !important;
    border-radius: 14px !important;
    background: #ffffff !important;
}
gr-button {
    background: #28a745 !important;
    color: white !important;
    border-radius: 8px !important;
}
gr-textbox {
    border: 1px solid #bbb !important;
}
""") as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("<h2 style='text-align:center;'>💸 FinGenius – Financial Planner for Pakistan</h2>")
        chatbot = gr.Chatbot(height=400, type="messages", show_copy_button=True)
        state = gr.State([])

        with gr.Row():
            txt = gr.Textbox(
                placeholder="Enter your question in English...",
                show_label=False,
                lines=2
            )
            submit = gr.Button("Send")

        submit.click(
            fn=chat_with_memory_english,
            inputs=[txt, state],
            outputs=[chatbot, state, txt]  # 👈 Clear input field
        )

demo.launch()