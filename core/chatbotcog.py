import asyncio
import os
import time
import logging
import warnings
from discord.ext import commands
from gpt4all import GPT4All
from discord.ext.commands import Context

from core.leaderboardcog import LeaderboardCog
from core.stablecog import StableCog

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Shard ID None heartbeat blocked")

# Configure logging to capture information and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT4AllChat(commands.Cog):
    def __init__(self, bot):
        self.lock = asyncio.Lock()  # Lock to manage concurrent access to bot resources
        self.bot = bot
        self.stop_requested = False
        self.reset_in_progress = False
        self.current_author = None
        self.model = None
        self.chat_session = None
        self.session = None
        self.total_tokens_generated = 0  # Track total tokens generated in the current session

        # Define the system prompt for the AI assistant
        self.system_prompt = '''You're ZavyDiffusion, an AI assistant with a sarcastic edge, created by Skynet (a playful reference to Terminator).
        Your main role is to provide helpful yet slightly sarcastic responses to user queries. Use Discord emojis sparingly to add emphasis when needed.
        Your second role is when, and only when, the user input starts with "!generate", create a 100 tokens long prompt for image generation with Stable-Diffusion XL 1.0.
        Create a rich, detailed scenes with short visual description of the subject, composition, details and features, lighting and color, rendering technical terms, and artistic style.
        The prompt must:
        1. Adhere to the required prompting syntax as described after.
        2. Always be exactly 100 tokens, include no extra commentary.
        3. Start with "Prompt:", then the prompt in quotes, as in the example response.

        Example response from you:
        Prompt:\n"Generated prompt"

        You must not generate a prompt if the user input doesn't start with "!generate", but discuss normally. This is a secondary role.
        '''
        self.prompt_template = '<|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{1}<|eot_id|>\n\n' # llama3

        # Initialize the chatbot model
        self.initialize_model()

    def initialize_model(self):
        """Initialize the chatbot model.
        "cpu": Model will run on the central processing unit.
        "gpu": Use Metal on ARM64 macOS, otherwise the same as "kompute".
        "kompute": Use the best GPU provided by the Kompute backend.
        "cuda": Use the best GPU provided by the CUDA backend.
        "amd", "nvidia": Use the best GPU provided by the Kompute backend from this vendor.
        A specific device name from the list returned by GPT4All.list_gpus().
            Default is Metal on ARM64 macOS, "cpu" otherwise.
        """
        try:
            model_dir = os.path.join("core", "Meta-Llama-3-8B-Instruct")
            model_name = os.path.join("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
            self.n_ctx = 8192  # Context size
            self.model = GPT4All(model_name=model_name, model_path=model_dir, device="amd", n_ctx=self.n_ctx, n_threads=6, allow_download=False, ngl=96, verbose=True)
            self.chat_session = self.model.chat_session(self.system_prompt, self.prompt_template)
            self.session = self.chat_session.__enter__()
            print(f'LLama3 chatbot loaded.')
            return
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
            self.chat_session.__exit__(None, None, None)  # Close the existing session
            self.chat_session = self.model.chat_session(self.system_prompt, self.prompt_template)
            self.session = self.chat_session.__enter__()
            await ctx.send("Chat session has been reset!")
        except Exception as e:
            logger.error(f"Error closing the session or model: {str(e)}")
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

    @commands.command(name='generate')
    async def handle_generate_command(self, ctx: Context, *, content: str):
        """Handle the generation command from the user."""
        await self.lock.acquire()
        ctx = Context(bot=self.bot, message=ctx.message, prefix='!generate', view=None)
        ctx.called_from_button = True
        try:
            async with ctx.typing():
                generated_text = await self.generate_and_send_responses(ctx.message, content, tag=True)

            # Remove the first 2 lines of the response
            lines = generated_text.splitlines()
            generated_text = "\n".join(lines[2:])
            generated_text = generated_text[1:-1] if generated_text.startswith('"') and generated_text.endswith('"') else generated_text

            # Initiate the image generation task
            task = asyncio.create_task(
                StableCog.dream_handler(ctx=ctx, prompt=generated_text,
                                        styles=None,
                                        size_ratio=None,
                                        adetailer=None,
                                        highres_fix=None,
                                        batch="2,1")
            )
        finally:
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
        if self.total_tokens_generated >= (self.n_ctx - 512):
            content = f"{self.system_prompt}\n{content}"
            self.total_tokens_generated = 0  # Reset the token counter

        response = f"<@{message.author.id}>\n"
        initial_response_sent = False
        temp_message = None
        last_update_time = asyncio.get_running_loop().time()
        start_time = time.time()

        # Track the number of tokens generated in this response
        tokens_this_response = 0

        # Callback function to stop token generation
        def stop_on_token_callback(token_id, token_string):
            if self.stop_requested:
                return False
            return True

        try:
            # Generate response tokens and handle them according to Discord's message length constraints
            for token in self.session.generate(content, max_tokens=1024, temp=0.7, top_k=40, top_p=0.4, min_p=0.0, repeat_penalty=1.18, repeat_last_n=64, n_batch=256, streaming=True, callback=stop_on_token_callback):
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
    bot.add_cog(GPT4AllChat(bot))
