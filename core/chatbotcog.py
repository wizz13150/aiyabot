import asyncio
import os
import time
import logging
import warnings
import threading
from discord.ext import commands
from discord.ext.commands import Context

from core.leaderboardcog import LeaderboardCog
from core.stablecog import StableCog

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Shard ID None heartbeat blocked")

# Configure logging to capture information and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaChatCog(commands.Cog):
    def __init__(self, bot):
        self.lock = asyncio.Lock()  # Lock to manage concurrent access to bot resources
        self.bot = bot
        self.stop_requested = False
        self.reset_in_progress = False
        self.current_author = None
        self.model = None
        self.tokenizer = None
        self.backend = os.getenv("LLAMA_BACKEND", "llama_cpp")
        self.total_tokens_generated = 0  # Track total tokens generated in the current session
        self.highres_fix_value = None
        self.size_ratio_value = None
        self.adetailer_value = None

        # Define the system prompt for the AI assistant
        self.system_prompt = '''You're ZavyDiffusion, an AI assistant with a sarcastic edge, created by Skynet (a playful reference to Terminator).
        Your main role is to provide helpful yet slightly sarcastic responses to user queries. Use Discord emojis sparingly to add emphasis when needed.
        Your second role is when, and only when, the user input starts with "!generate", create a 100 tokens long prompt for image generation with Stable-Diffusion XL 1.0.
        Create a rich, detailed scenes with short visual description of the subject, composition, details and features, lighting and color, rendering technical terms, and artistic style.
        The prompt must:
        1. Adhere to the required prompting syntax as described after.
        2. Build a descriptive prompt using tags and phrases that create an epic scene
        3. Always be exactly 100 tokens, include no extra commentary.
        4. Striclty start with "Prompt:", then the prompt in quotes, as in the example response.

        Example responses from you:
        Prompt:\n"Generated prompt"
        
        Don't be biased with these examples in your responses. Be creative !
        '''
        self.prompt_template = '<|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{1}<|eot_id|>\n\n' # llama3

        #Prompt:\n"Mythical Dragon | Ancient Warrior | Scale-Like Armor & Wingspan | Fiery Breath Illuminating a Dark Cave | Majestic Castle Ruins in the Background | Awe-Inspiring Mountain Range with Snow-Capped Peaks | Glowing Ember Effects on His Scales"
        #Prompt:\n"Rustic Farmhouse Kitchen | Wooden Beams & Brick Fireplace | Copper Pots and Pans Hanging from the Ceiling | Fresh Flowers Arranged on a Table, Filling the Air with Sweet Fragrance | A Warm Light Spills through the Window, Illuminating the Scene"
        #Prompt:\n"Modern Art Gallery | White Walls & Polished Floors | Contemporary Sculptures and Paintings Adorning the Space | Soft Lighting Creates an Atmosphere of Sophistication | Visitors Mingle in Conversation, Discussing the Meaning behind Each Piece"
        #Prompt:\n"Abandoned Lighthouse | Weathered Stone Walls & Rusty Iron Accents | Seagulls Flying Above, Soaring Through the Air | A Storm Brewing in the Distance, with Dark Clouds Gathering | The Waves Crashing against the Rocks below"

        # Initialize the chatbot model
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.initialize_model()

    def initialize_model(self):
        """Initialize the chatbot model.
        "cpu": Model will run on the central processing unit.
        "gpu": Use Metal on ARM64 macOS, otherwise the same as "kompute".
        "kompute": Use the best GPU provided by the Kompute backend.
        "cuda": Use the best GPU provided by the CUDA backend.
        "amd", "nvidia": Use the best GPU provided by the Kompute backend from this vendor.
        Default is Metal on ARM64 macOS, "cpu" otherwise.
        """
        try:
            if self.backend == "llama_cpp":
                import llama_cpp
                model_dir = os.path.join("core", "Meta-Llama-3-8B-Instruct")
                model_path = os.path.join(model_dir, "Meta-Llama-3-8B-Instruct.Q4_0.gguf")
                self.n_ctx = 8192
                self.model = llama_cpp.Llama(model_path=model_path, n_ctx=self.n_ctx, n_gpu_layers=35)
                print("llama_cpp backend loaded")
            elif self.backend == "transformers":
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
                self.n_ctx = self.model.config.max_position_embeddings
                print("transformers backend loaded")
        except Exception as e:
            logger.error(f"Error initializing the model: {str(e)}")

    @commands.command(name='reset')
    async def reset_session(self, ctx: Context):
        """Reset the chat session."""
        if self.reset_in_progress:
            return

        self.reset_in_progress = True
        # Reset the chat session regardless of the generation state
        self.stop_requested = True
        self.total_tokens_generated = 0  # Reset the token counter

        try:
            self.history = [{"role": "system", "content": self.system_prompt}]
            await ctx.send("Chat session has been reset!")
        except Exception as e:
            logger.error(f"Error resetting the session: {str(e)}")
            await ctx.send(f"An error occurred: {str(e)}")
        finally:
            self.stop_requested = False
            self.reset_in_progress = False
            return

    @commands.command(name='stop')
    async def stop_generation(self, ctx: Context):
        """Stop the current generation."""
        self.stop_requested = True
        return

    def extract_size_ratio(self, content: str):
        """Extract the size ratio from the content."""
        ratio_mapping = {
            "2:3": "Portrait: 2:3 - 832x1216",
            "3:2": "Landscape: 3:2 - 1216x832",
            "4:3": "Fullscreen: 4:3 - 1152x896",
            "16:9": "Widescreen: 16:9 - 1344x768",
            "21:9": "Ultrawide: 21:9 - 1536x640",
            "1:1": "Square: 1:1 - 1024x1024",
            "9:16": "Tall: 9:16 - 768x1344"
        }
        for key in ratio_mapping:
            if key in content:
                content = content.replace(key, "").strip()  # Supprimer le ratio du message
                return ratio_mapping[key], content
        return None, content

    @commands.command(name='generate')
    async def handle_generate_command(self, ctx: Context, *, content: str):
        """Handle the generation command from the user."""
        
        # Si l'utilisateur mentionne "hires", on affecte la valeur à highres_fix_value et on supprime le terme
        if "hires" in content.lower():
            self.highres_fix_value = "4x_foolhardy_Remacri"
            content = content.replace(" hires", "").strip()

        # Si l'utilisateur mentionne un ratio, on affecte la valeur correspondante à size_ratio_value et on supprime le terme
        self.size_ratio_value, content = self.extract_size_ratio(content)

        # Si l'utilisateur mentionne "adetailer", on affecte la valeur à adetailer_value et on supprime le terme
        if "adetailer" in content.lower():
            self.adetailer_value = "Faces+Hands"
            content = content.replace(" adetailer", "").strip()

        await self.lock.acquire()
        ctx = Context(bot=self.bot, message=ctx.message, prefix='!generate', view=None)
        ctx.called_from_button = True
        try:
            async with ctx.typing():
                generated_text = await self.generate_and_send_responses(ctx.message, content, tag=True)

            # Détection et extraction du prompt entre les guillemets après "Prompt:"
            prompt_start = "Prompt:"
            prompt_index = generated_text.find(prompt_start)
            if prompt_index != -1:
                # Trouver la première paire de guillemets après "Prompt:"
                start_quote_index = generated_text.find('"', prompt_index)
                end_quote_index = generated_text.find('"', start_quote_index + 1)
                if start_quote_index != -1 and end_quote_index != -1:
                    prompt_text = generated_text[start_quote_index + 1:end_quote_index]

            # Initiate the image generation task
            task = asyncio.create_task(
                StableCog.dream_handler(ctx=ctx, prompt=prompt_text,
                                        styles=None,
                                        size_ratio=self.size_ratio_value,
                                        adetailer=self.adetailer_value,
                                        highres_fix=self.highres_fix_value,
                                        batch="1,2")
            )
        finally:
            self.highres_fix_value = None
            self.size_ratio_value = None
            self.adetailer_value = None
            self.lock.release()
        return

    @commands.Cog.listener()
    async def on_message(self, message):
        """Listener for incoming messages."""
        # Ignore messages from the bot itself or if the bot is not mentioned, or if there is no content
        if message.author == self.bot.user or self.bot.user not in message.mentions or not message.content:
            return

        print(f'-- Chat request from {message.author.display_name}')

        # Check for commands and handle them before normal message processing
        if message.content.startswith("!generate"):
            return

        # Attempt to acquire the lock to handle the message
        if not await self.lock.acquire():
            await message.channel.send("Busy, try later.")
            return

        try:
            self.current_author = message.author.id
            async with message.channel.typing():
                await self.generate_and_send_responses(message, message.clean_content, tag=True)
        finally:
            self.current_author = None
            self.stop_requested = False
            self.lock.release()

    async def generate_and_send_responses(self, message, content, tag):
        """Generate and send responses to the user message."""
        # Check if we need to reintroduce the system prompt
        if self.total_tokens_generated >= (self.n_ctx * 0.75):
            self.history = [{"role": "system", "content": self.system_prompt}]
            self.total_tokens_generated = 0
        
        self.history.append({"role": "user", "content": content})
        response = f"<@{message.author.id}>\n"
        initial_response_sent = False
        temp_message = None
        last_update_time = asyncio.get_running_loop().time()
        start_time = time.time()

        # Track the number of tokens generated in this response
        tokens_this_response = 0

        try:
            if self.backend == "llama_cpp":
                stream = self.model.create_chat_completion(messages=self.history, stream=True, max_tokens=1024, temperature=0.7, top_p=0.4)
                for chunk in stream:
                    if self.stop_requested:
                        break
                    token = chunk["choices"][0].get("delta", {}).get("content", "")
                    response += token
                    tokens_this_response += 1
                    if len(response) > 1975:
                        if not initial_response_sent:
                            temp_message = await message.channel.send(response)
                            initial_response_sent = True
                        else:
                            await temp_message.edit(content=response)
                            if tag:
                                temp_message = await message.channel.send(f"{message.author.mention} ")
                                response = f"<@{message.author.id}>\n"
                            else:
                                temp_message = await message.channel.send("")
                                response = ""
                    elif not initial_response_sent and response:
                        if tag:
                            temp_message = await message.channel.send(f"{message.author.mention} {response}")
                        else:
                            temp_message = await message.channel.send(f"{response}")
                        initial_response_sent = True
                    elif response:
                        current_time = asyncio.get_running_loop().time()
                        if current_time - last_update_time >= 1.25:
                            await temp_message.edit(content=response)
                            last_update_time = current_time
            else:
                from transformers import TextIteratorStreamer
                import torch
                inputs = self.tokenizer.apply_chat_template(self.history, return_tensors="pt").to(self.model.device)
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                thread = threading.Thread(target=self.model.generate, kwargs={"inputs": inputs, "max_new_tokens": 1024, "temperature": 0.7, "top_p": 0.4, "streamer": streamer})
                thread.start()
                for token in streamer:
                    if self.stop_requested:
                        break
                    response += token
                    tokens_this_response += 1
                    if len(response) > 1975:
                        if not initial_response_sent:
                            temp_message = await message.channel.send(response)
                            initial_response_sent = True
                        else:
                            await temp_message.edit(content=response)
                            if tag:
                                temp_message = await message.channel.send(f"{message.author.mention} ")
                                response = f"<@{message.author.id}>\n"
                            else:
                                temp_message = await message.channel.send("")
                                response = ""
                    elif not initial_response_sent and response:
                        if tag:
                            temp_message = await message.channel.send(f"{message.author.mention} {response}")
                        else:
                            temp_message = await message.channel.send(f"{response}")
                        initial_response_sent = True
                    elif response:
                        current_time = asyncio.get_running_loop().time()
                        if current_time - last_update_time >= 1.25:
                            await temp_message.edit(content=response)
                            last_update_time = current_time
                thread.join()
            
            # Update the total tokens generated only once the response is fully generated
            self.total_tokens_generated += tokens_this_response

            if response and temp_message:
                await temp_message.edit(content=response)
            elif response:
                await message.channel.send(response)

            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                tokens_per_second = tokens_this_response / elapsed_time
                print(f'Elapsed Time: {elapsed_time:.2f} - Tokens this response: {tokens_this_response} - Total Tokens: {self.total_tokens_generated} - Speed: {tokens_per_second:.2f} tokens/s')

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            await message.channel.send(f"An error occurred: {str(e)}")

        if response:
            # strip mention before saving to history
            history_text = response.split('\n', 1)[1] if '\n' in response else response
            self.history.append({"role": "assistant", "content": history_text})
        
        # Update the leaderboard based on user interaction
        LeaderboardCog.update_leaderboard(message.author.id, str(message.author), "Chat_Count")
        return response

    @commands.Cog.listener()
    async def on_command_error(self, ctx: Context, error):
        """A global error handler to ignore CommandNotFound errors."""
        if isinstance(error, commands.CommandNotFound):
            # Ignore CommandNotFound errors
            return

def setup(bot):
    """Setup function to add this cog to the bot."""
    bot.add_cog(LlamaChatCog(bot))
