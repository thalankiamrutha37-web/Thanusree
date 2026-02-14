from google import genai
import gradio as gr
from gtts import gTTS
import tempfile
import os

# Keep your API key & client unchanged
client = genai.Client(api_key="YOUR_API_KEY")

# AI Fashion Advisor Bot
def fashion_bot(user_text, user_image, user_audio):
    contents = []

    # Convert audio to text (Speech-to-Text)
    if user_audio is not None:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(user_audio) as source:
            audio_data = recognizer.record(source)
            try:
                user_text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                user_text = "Sorry, I could not understand your voice input."

    if user_text:
        contents.append(
            f"You are a professional AI Fashion Designer and Stylist. "
            f"Provide personalized fashion recommendations based on the user's request. "
            f"Suggest outfit combinations, color matching, fabric choices, accessories, "
            f"footwear, seasonal trends, and styling tips.\n\n"
            f"User Input:\n{user_text}"
        )

    if user_image is not None:
        contents.append(user_image)  # Dress/outfit image

    # Call Gemini model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )

    bot_reply = response.text

    # Convert reply to speech (Text-to-Speech)
    tts = gTTS(bot_reply)
    tmp_path = tempfile.mktemp(suffix=".mp3")
    tts.save(tmp_path)

    return bot_reply, tmp_path


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 👗 Generative AI Powered Fashion Design Recommendation System")
    gr.Markdown(
        "✨ Ask about outfit ideas, color combinations, seasonal trends, or styling tips.\n"
        "📷 Upload outfit images for fashion analysis.\n"
        "🎤 Speak your fashion question with voice input.\n"
        "🔊 Get AI fashion advice in both text and voice."
    )

    with gr.Row():
        with gr.Column():
            user_text = gr.Textbox(
                label="Enter your fashion query",
                placeholder="Example: Suggest an outfit for a summer wedding"
            )
            user_image = gr.Image(label="Upload Outfit Image (optional)", type="filepath")
            user_audio = gr.Audio(label="Speak your question (optional)", type="filepath")
            submit_btn = gr.Button("Get AI Fashion Advice")

        with gr.Column():
            output_text = gr.Textbox(label="AI Fashion Designer's Recommendation", interactive=False, lines=12)
            output_audio = gr.Audio(label="AI Voice Response", type="filepath")

    submit_btn.click(
        fn=fashion_bot,
        inputs=[user_text, user_image, user_audio],
        outputs=[output_text, output_audio]
    )

# Run chatbot
if __name__ == "__main__":
    demo.launch()
