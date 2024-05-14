import asyncio
import time
import logging
from discord.ext import commands
from gpt4all import GPT4All
from discord.ext.commands import Context

from core.leaderboardcog import LeaderboardCog
from core.stablecog import StableCog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT4AllChat(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.lock = asyncio.Lock()  # Lock to manage concurrent access to bot resources
        self.stop_requested = False
        self.current_author = None
        self.model = None
        self.chat_session = None
        self.session = None
        self.system_prompt = 'You are an AI assistant named ZavyDiffusion that follows instruction extremely well. Your primary goal is to help as much as you can. Be slightly sarcastic when possible without altering the response quality.\nWhen a user command begins with "!generate", your second role is to generate a prompt suitable for Stable-Diffusion. This prompt should strictly adhere to the syntax required for image generation, consist of 200 tokens, and contain no additional commentary or text.\n'
        self.prompt_template = '<|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{1}<|eot_id|>\n\n' # llama3
        self.initialize_model()

    def initialize_model(self):
        try:
            self.model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="amd", n_ctx=8192, n_threads=6, allow_download=True, ngl=96)
            self.chat_session = self.model.chat_session(self.system_prompt, self.prompt_template)
            self.session = self.chat_session.__enter__()
            print(f'LLama3 chatbot loaded. Session:\n{self.session}')
        except Exception as e:
            logger.error(f"Error initializing the model: {str(e)}")

    async def reset_session(self):
        print(f'Resetting session context...')
        if self.model:
            try:
                print(f'Close current model instance...')
                self.model.close()
            except Exception as e:
                logger.error(f"Error closing the model: {str(e)}")

        self.initialize_model()

    async def get_context_from_message(self, message):
        ctx = Context(bot=self.bot, message=message, prefix='!', view=None)
        ctx.called_from_button = True
        return ctx

    async def handle_generate_command(self, message, content):
        # Generate the prompt/response
        async with message.channel.typing():
            generated_text = await self.generate_and_send_responses(message, content, tag=False)

        # Delete 3 first lines of the response
        lines = generated_text.splitlines()
        generated_text = "\n".join(lines[3:])
        generated_text = generated_text[1:-1] if generated_text.startswith('"') and generated_text.endswith('"') else generated_text

        self.author = message.author
        self.channel = message.channel
        self.called_from_button = True

        # Create a context object
        ctx = await self.get_context_from_message(message)

        # Generate the pic from the response
        task = asyncio.create_task(
            StableCog.dream_handler(ctx=ctx, prompt=generated_text,
                                    styles=None,
                                    size_ratio=None,
                                    adetailer=None,
                                    highres_fix=None,
                                    batch="2,1")
        )
        self.lock.release()

    @commands.Cog.listener()
    async def on_message(self, message):
        # Stop generation if the "!stop" command is detected
        if message.content.lower() == "!stop" and message.author.id == self.current_author:
            self.stop_requested = True
            return

        # Reset chat_session if the "!reset" command is detected
        #if message.content.lower() == "!reset" and message.author.id == self.current_author:
        #    await self.reset_session()
        #    await message.channel.send("Chat session has been reset.")
        #    return

        # Clean the message content to remove unnecessary mentions
        content = message.clean_content.replace(f'@{self.bot.user.display_name}', '').strip()
        # Ignore messages from the bot itself or if the bot is not mentioned, or if there is no content
        if message.author == self.bot.user or self.bot.user not in message.mentions or not content:
            return

        print(f'-- Chat request from {message.author.display_name}')

        # Attempt to acquire the lock to handle the message
        if not await self.lock.acquire():
            await message.channel.send("Busy, try later.")
            return

        # Detect if the message asks for an image
        if content.lower().startswith("!generate"):
            await self.handle_generate_command(message, content)
            return

        try:
            self.current_author = message.author.id
            async with message.channel.typing():
                await self.generate_and_send_responses(message, content, tag=True)
        finally:
            self.current_author = None
            self.stop_requested = False
            self.lock.release()

    async def generate_and_send_responses(self, message, content, tag):
        if self.stop_requested:
            return

        response = f"<@{message.author.id}>\n"
        initial_response_sent = False
        temp_message = None
        last_update_time = asyncio.get_running_loop().time()
        start_time = time.time()
        token_count = 0

        # Callback function to stop token generation
        def stop_on_token_callback(token_id, token_string):
            if self.stop_requested and '.' in token_string:
                return False
            return True

        try:
            # Generate response tokens and handle them according to Discord's message length constraints
            for token in self.session.generate(content, max_tokens=1024, temp=0.7, top_k=40, top_p=0.4, min_p=0.0, repeat_penalty=1.18, repeat_last_n=64, n_batch=128, streaming=True, callback=stop_on_token_callback):
                response += token
                token_count += 1
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

            if self.stop_requested:
                return

            if response and temp_message:
                await temp_message.edit(content=response)
            elif response:
                await message.channel.send(response)

            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                tokens_per_second = token_count / elapsed_time
                print(f'Elapsed Time: {elapsed_time:.2f} - Tokens: {token_count} - Speed: {tokens_per_second:.2f} tokens/s')

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            await message.channel.send(f"An error occurred: {str(e)}")

        # Update the leaderboard based on user interaction
        LeaderboardCog.update_leaderboard(message.author.id, str(message.author), "Chat_Count")
        return response

def setup(bot):
    # Setup function to add this cog to the bot
    bot.add_cog(GPT4AllChat(bot))
