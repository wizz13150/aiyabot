import asyncio
import time

from discord.ext import commands
from gpt4all import GPT4All

from core.leaderboardcog import LeaderboardCog
from core.stablecog import StableCog

class GPT4AllChat(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.lock = asyncio.Lock()  # Lock to manage concurrent access to bot resources
        self.stop_requested = False
        self.current_author = None
        self.model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="amd", n_ctx=8192, n_threads=6, allow_download=True, ngl=96)
        self.system_prompt = 'You are an AI assistant named ZavyDiffusion that follows instruction extremely well. Your primarey goal is to help as much as you can. Be slightly sarcastic when possible without altering the response quality\nWhen a user command begins with "!generate", your second role is to generate a prompt suitable for Stable-Diffusion. This prompt should strictly adhere to the syntax required for image generation, consist of 200 tokens, and contain no additional commentary or text.\n'
        self.prompt_template = '<|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{1}<|eot_id|>\n\n' # llama3
        self.chat_session = self.model.chat_session(self.system_prompt, self.prompt_template)
        self.session = self.chat_session.__enter__()
        print(f'Chatbot loaded')

    @commands.Cog.listener()
    async def on_message(self, message):
        # Stop generation if the "!stop" command is detected
        if message.content == "!stop" and message.author.id == self.current_author:
            self.stop_requested = True
            return

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
        if content.startswith("!generate"):
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
            ctx = self
            # Generate the pic from the response
            task = asyncio.create_task(
                StableCog.dream_handler(ctx=ctx, prompt=generated_text,
                                        styles=None,
                                        size_ratio=None,
                                        adetailer=None,
                                        highres_fix=None,
                                        batch="2,1")
            )
            #task.add_done_callback(self.handle_dream_completion)
            self.lock.release()
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
        print(f"tag:{tag}")
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
            for token in self.session.generate(content, max_tokens=4096, temp=0.7, top_k=40, top_p=0.4, min_p=0.0, repeat_penalty=1.18, repeat_last_n=64, n_batch=128, streaming=True, callback=stop_on_token_callback):
                response += token
                token_count += 1
                if len(response) > 1975:
                    if not initial_response_sent:
                        temp_message = await message.channel.send(response)
                        initial_response_sent = True
                    else:
                        await temp_message.edit(content=response)
                        if tag is True:
                            temp_message = await message.channel.send(f"{message.author.mention} ")
                            response = f"<@{message.author.id}>\n"
                        else:
                            temp_message = await message.channel.send("")
                            response = ""
                elif not initial_response_sent and response:
                    if tag is True:
                        temp_message = await message.channel.send(f"{message.author.mention} {response}")
                    else:
                        temp_message = await message.channel.send(f"{response}")
                    initial_response_sent = True
                elif response:
                    current_time = asyncio.get_running_loop().time()
                    if current_time - last_update_time >= 1.25:
                        await temp_message.edit(content=response)
                        last_update_time = current_time

            if self.stop_requested is True:
                return

            if response:
                await temp_message.edit(content=response)

            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                tokens_per_second = token_count / elapsed_time
                print(f'Elapsed Time: {elapsed_time} - Tokens: {token_count} - Speed: {tokens_per_second:.2f} tokens/s')

        except Exception as e:
            ...

        # Update the leaderboard based on user interaction
        LeaderboardCog.update_leaderboard(message.author.id, str(message.author), "Chat_Count")
        return response

def setup(bot):
    # Setup function to add this cog to the bot
    chat_cog = GPT4AllChat(bot)
    bot.add_cog(chat_cog)
