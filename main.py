import os
import asyncio
import base64
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file")

@cl.on_chat_start
async def start():

    #Reference: https://ai.google.dev/gemini-api/docs/openai
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )
    
    cl.user_session.set("chat history", [])
    cl.user_session.set("config", config)

    # Enhanced instructions for file handling including images
    agent: Agent = Agent(
        name="Assistant", 
        instructions="""You are a helpful assistant that can read and analyze uploaded files including text files, PDFs, documents, code files, and IMAGES. 
When users upload image files, you can analyze and describe the visual content, extract text from images, and provide insights about the images.
For text-based files, read the content and provide helpful responses.
For image files, describe what you see and answer questions about the visual content.""", 
        model=model
    )

    cl.user_session.set("agent", agent)
    await cl.Message(content="Welcome to the Panaversity AI Assistant! How can I help you today? You can upload files (including images) and I'll read and analyze them for you.").send()

    @cl.on_message
    async def main(message: cl.Message):
        msg = cl.Message(content="Thinking...")
        await msg.send()

        agent: Agent = cast(Agent, cl.user_session.get("agent"))
        config: RunConfig = cast(RunConfig, cl.user_session.get("config"))
        history = cl.user_session.get("chat history") or []
        
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
                                # For Gemini models, we need to format the image properly
                                file_content += f"\n\n[Image File: {getattr(element, 'name', 'unknown')} - Base64 encoded image data included for analysis]"
                                # Add image to history in a format the model can understand
                                history.append({
                                    "role": "user", 
                                    "content": [
                                        {"type": "text", "text": f"User uploaded an image file: {getattr(element, 'name', 'unknown')}. Please analyze this image."},
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                                    ]
                                })
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

        # Combine user message with file content (for text files)
        full_message = message.content
        if file_content:
            full_message += file_content

        # Only add text message to history if we didn't already add image content
        if not any(isinstance(item, dict) and 'image_url' in str(item) for item in history):
            history.append({"role": "user", "content": full_message})

        try:
            print('\n[CALLING_AGENT_WITH_CONTEXT]\n', history, "\n")

            result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: Runner.run_sync(
                starting_agent=agent,
                input=history,
                run_config=config
            )
        )

            response_content = result.final_output
            msg.content = response_content
            await msg.update()

            cl.user_session.set("chat history", result.to_input_list())

            print(f"User: {message.content}")
            print(f"Assistant: {response_content}")

        except Exception as e:
            msg.content = f"Error: {str(e)}"
            await msg.update()
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # This allows Render to set the port
    port = int(os.environ.get("PORT", 8000))
    cl.run(app, host="0.0.0.0", port=port)