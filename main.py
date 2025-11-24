import os
import asyncio
import base64
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from openai import AsyncOpenAI

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file")

@cl.on_chat_start
async def start():
    # Reference: https://ai.google.dev/gemini-api/docs/openai
    client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    
    cl.user_session.set("client", client)
    cl.user_session.set("chat_history", [])
    
    await cl.Message(content="Welcome to the Panaversity AI Assistant! How can I help you today? You can upload files (including images) and I'll read and analyze them for you.").send()

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="Thinking...")
    await msg.send()

    client: AsyncOpenAI = cast(AsyncOpenAI, cl.user_session.get("client"))
    history = cl.user_session.get("chat_history") or []
    
    # Process file attachments if any
    file_content = ""
    if hasattr(message, 'elements') and message.elements:
        for element in message.elements:
            try:
                # Check if it's an image file
                file_name = getattr(element, 'name', 'unknown').lower()
                if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    # Process image file - encode as base64 for the model
                    if hasattr(element, 'content') and element.content:
                        # Convert image to base64
                        image_data = element.content
                        if isinstance(image_data, bytes):
                            base64_image = base64.b64encode(image_data).decode('utf-8')
                            file_content += f"\n\n[Image File: {getattr(element, 'name', 'unknown')} - Base64 encoded image data included for analysis]"
                            await cl.Message(content=f"🖼️ Processing image: {getattr(element, 'name', 'unknown')}").send()
                        else:
                            file_content += f"\n\n[Image File: {getattr(element, 'name', 'unknown')} - Content not in bytes format]"
                    else:
                        file_content += f"\n\n[Image File: {getattr(element, 'name', 'unknown')} - No content available]"
                else:
                    # Process text-based files
                    if hasattr(element, 'content'):
                        file_text = element.content
                        if isinstance(file_text, bytes):
                            file_text = file_text.decode('utf-8')
                        file_content += f"\n\n[File: {getattr(element, 'name', 'unknown')}]\n{file_text}"
                        await cl.Message(content=f"📁 Read file: {getattr(element, 'name', 'unknown')}").send()
            except Exception as e:
                file_content += f"\n\n[Error reading file {getattr(element, 'name', 'unknown')}: {str(e)}]"

    # Combine user message with file content
    full_message = message.content
    if file_content:
        full_message += file_content

    # Add to history
    history.append({"role": "user", "content": full_message})

    try:
        # Call Gemini API directly
        response = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=history,
            max_tokens=1000
        )
        
        response_content = response.choices[0].message.content
        msg.content = response_content
        await msg.update()

        # Add assistant response to history
        history.append({"role": "assistant", "content": response_content})
        cl.user_session.set("chat_history", history)

        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")

    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    cl.run(app, host="0.0.0.0", port=port)