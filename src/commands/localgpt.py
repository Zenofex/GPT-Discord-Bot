#import discord
import json
import requests

#from discord import app_commands
from bot import discord, app_commands
from bot import client

from ollama import Client, ResponseError

OLLAMA_HOST = 'http://ollama:11434'
client_ollama = Client(host=OLLAMA_HOST)  # Ollama client instance

async def get_ollama_models():
    try:
        available_models = client_ollama.list()
        models = [model['name'] for model in available_models['models']]
        if len(models) == 0:
            models.append("llama3")
        print(f"[I] Available models: {models}")
        return models
    except Exception as e:
        print(f"[E] Error fetching models from Ollama: {e}")
        return []

async def ensure_model_available(model):
    available_models = await get_ollama_models()
    if model not in available_models:
        print(f"[I] Model '{model}' not found. Pulling...")

        # Run the blocking pull operation in a thread to avoid blocking the event loop
        def pull_model():
            try:
                client_ollama.pull(model)
                print(f"[I] Successfully pulled model '{model}'.")
            except Exception as e:
                print(f"[E] Failed to pull model '{model}': {e}")
                raise e
        
        # Offload the pull to a background thread
        await asyncio.to_thread(pull_model)

# Autocomplete for dynamic model selection in Discord
async def model_autocomplete(interaction: discord.Interaction, current: str):
    models = await get_ollama_models()
    return [
        app_commands.Choice(name=model, value=model)
        for model in models if current.lower() in model.lower()
    ][:25]  # Limit to 25 results

@client.tree.command(description='Receive a text response from a local Ollama model.')
@app_commands.describe(
    prompt='Text prompt to pass to the model for a response.',
    model="The Ollama model to use"
)
@app_commands.autocomplete(model=model_autocomplete)
async def local_gpt(
    message_ctx: discord.Interaction,
    prompt: str,
    model: str
):
    print(f"[I] Received local_gpt prompt: {prompt}")
    await message_ctx.response.defer()

    if not model:
        models = await get_ollama_models()
        if not models:
            await message_ctx.followup.send("No models available on Ollama server.")
            return
        model = models[0]  # Use first available model if none selected

    await ensure_model_available(model)

    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.playing, name="thinking..."))

    # Generate response from Ollama using client.chat
    message_list = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client_ollama.chat(model=model, messages=message_list)
    except ResponseError as e:
        if "not found" in str(e):  # If the model isn't loaded, pull it
            print(f"[W] Model '{model}' not found. Pulling now...")

            # Offload the model pull to avoid blocking
            await asyncio.to_thread(lambda: client_ollama.pull(model))
            
            print(f"[I] Model '{model}' pulled successfully. Retrying request...")
            response = client_ollama.chat(model=model, messages=message_list)
        else:
            raise e  # If it's another error, re-raise

    if response and 'message' in response and 'content' in response['message']:
        bot_response = response['message']['content']
        await message_ctx.followup.send(bot_response)
    else:
        await message_ctx.followup.send("Failed to generate response from Ollama.")

    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="Ready for prompts."))                                                        
