from discord.ext import commands
from gpt4all import GPT4All
import asyncio

class GPT4AllChat(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.lock = asyncio.Lock()

        #self.model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf", device="amd", n_ctx=2048, n_threads=4, allow_download=True, ngl=26)
        self.model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", device="amd", n_ctx=16096, n_threads=8, allow_download=False, ngl=32)
        #self.model = GPT4All("Llama-3-8B-Instruct-Gradient-1048k.Q4_0.gguf", device="amd", n_ctx=16000, n_threads=8, allow_download=False, ngl=32)
        #self.model = GPT4All("Phi-3-mini-4k-instruct.Q4_0", device="amd", n_ctx=2048, n_threads=4, allow_download=True, ngl=32)

        #system_prompt = '### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n'
        #self.system_prompt = "You are an AI assistant named 'Dr. ZavyPunk' and you must help the user as much as you can. You act as an expert in all domains, using best practice and ingenious logic.\n\n"
        self.system_prompt = ''
        #prompt_template = '### User:\n{0}\n\n### Response:\n'     # orca
        self.prompt_template = '<|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{1}<|eot_id|>\n\n' # llama3
        #prompt_template = '<|user|>\n{0}<|end|>\n<|assistant|>\n{1}<|end|>'     # phi3
        
        self.chat_session = self.model.chat_session(self.system_prompt, self.prompt_template)  # Create a persistent chat session
        print(f'Chatbot loaded')
        
        # Enter the chat session. Single session for now, for all channels and users.
        self.session = self.chat_session.__enter__()
        print(f'Enter ChatSession')


    @commands.Cog.listener()
    async def on_message(self, message):
        content = message.clean_content.replace(f'@{self.bot.user.display_name}', '').strip()

        # manage !reset
        if content == "!reset":
            print("Reset command received, resetting chat session...")
            self.session.__exit__(None, None, None)
            self.session = self.chat_session.__enter__()
            await message.channel.send(f"{message.author.mention} Session has been reset.")
            return

        if message.author == self.bot.user or self.bot.user not in message.mentions:
            return
        if not content:
            return
        if not await self.lock.acquire():
            await message.author.send("Busy, try later.")
            return

        print(f'-- Chat request from {message.author.display_name}')
        print(f'-- ChatSession: {self.chat_session}')

        try:
            async with message.channel.typing():
                await self.generate_and_send_responses(message, content)
        finally:
            self.lock.release()

    async def generate_and_send_responses(self, message, content):
        response = f"<@{message.author.id}>\n"
        initial_response_sent = False
        temp_message = None
        last_update_time = asyncio.get_running_loop().time()

        try:
            for token in self.session.generate(content, max_tokens=1024, temp=0.7, top_k=40, top_p=0.4, min_p=0.0, repeat_penalty=1.18, repeat_last_n=64, n_batch=512, streaming=True):
                response += token
                if len(response) > 1975:
                    if not initial_response_sent:
                        temp_message = await message.channel.send(response)
                        initial_response_sent = True
                    else:
                        await temp_message.edit(content=response)
                        temp_message = await message.channel.send(f"{message.author.mention} ")
                        response = f"<@{message.author.id}>\n"
                elif not initial_response_sent and response:
                    temp_message = await message.channel.send(f"{message.author.mention} {response}")
                    initial_response_sent = True
                elif response:
                    current_time = asyncio.get_running_loop().time()
                    if current_time - last_update_time >= 1.25:
                        await temp_message.edit(content=response)
                        last_update_time = current_time

            if response:
                await temp_message.edit(content=response)
        except Exception as e:
            ...

def setup(bot):
    chat_cog = GPT4AllChat(bot)
    bot.add_cog(chat_cog)
