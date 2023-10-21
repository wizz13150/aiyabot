import discord
import traceback
from asyncio import AbstractEventLoop, get_event_loop, run_coroutine_threadsafe
from discord import option
from discord.ui import Button, View
from discord.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from typing import Optional

from core import queuehandler
from core import settings
from core.queuehandler import GlobalQueue
from core.stablecog import StableCog
from core.leaderboardcog import LeaderboardCog


class PortraitButton(Button):
    def __init__(self, parent_view):
        super().__init__(label="Portrait", custom_id="portrait", emoji="🖼️", style=1)  # Use 1 for primary color
        self.parent_view = parent_view

    async def callback(self, interaction):
        self.style = 3  # Set this button to green
        for item in self.parent_view.children:
            if isinstance(item, LandscapeButton):
                item.style = 1  # Set the other button to default color
        self.parent_view.selected_orientation = "Portrait"
        await interaction.response.edit_message(view=self.parent_view)


class LandscapeButton(Button):
    def __init__(self, parent_view):
        super().__init__(label="Landscape", custom_id="landscape", emoji="🌅", style=1)
        self.parent_view = parent_view

    async def callback(self, interaction):
        self.style = 3  # Set this button to green
        for item in self.parent_view.children:
            if isinstance(item, PortraitButton):
                item.style = 1  # Set the other button to default color
        self.parent_view.selected_orientation = "Landscape"
        await interaction.response.edit_message(view=self.parent_view)


class NormalButton(Button):
    def __init__(self, parent_view):
        super().__init__(label="Normal", custom_id="normal", emoji="🔳", style=1)
        self.parent_view = parent_view

    async def callback(self, interaction):
        self.style = 3  # Set this button to green
        for item in self.parent_view.children:
            if isinstance(item, BigButton):
                item.style = 1  # Set the other button to default color
        self.parent_view.selected_size = "Medium"
        await interaction.response.edit_message(view=self.parent_view)


class BigButton(Button):
    def __init__(self, parent_view):
        super().__init__(label="Big", custom_id="big", emoji="⬛", style=1)
        self.parent_view = parent_view

    async def callback(self, interaction):
        self.style = 3  # Set this button to green
        for item in self.parent_view.children:
            if isinstance(item, NormalButton):
                item.style = 1  # Set the other button to default color
        self.parent_view.selected_size = "Big"
        await interaction.response.edit_message(view=self.parent_view)


class ChromaButton(Button):
    def __init__(self, parent_view):
        super().__init__(label="ZavyChromaXL", custom_id="chroma", emoji="✨", style=1)
        self.parent_view = parent_view

    async def callback(self, interaction):
        self.style = 3  # Set this button to green
        for item in self.parent_view.children:
            if isinstance(item, YumeButton):
                item.style = 1  # Set the other button to default color
        self.parent_view.selected_model = "ZavyChromaXL"
        await interaction.response.edit_message(view=self.parent_view)


class YumeButton(Button):
    def __init__(self, parent_view):
        super().__init__(label="ZavyYumeXL", custom_id="yume", emoji="🌟", style=1)
        self.parent_view = parent_view

    async def callback(self, interaction):
        self.style = 3  # Set this button to green
        for item in self.parent_view.children:
            if isinstance(item, ChromaButton):
                item.style = 1  # Set the other button to default color
        self.parent_view.selected_model = "ZavyYumeXL"
        await interaction.response.edit_message(view=self.parent_view)


class FaceDetailerButton(Button):
    def __init__(self, parent_view):
        super().__init__(label="Face Details", custom_id="adetailer", emoji="🎭", style=1)
        self.parent_view = parent_view

    async def callback(self, interaction):
        if self.style == 1:  # Default color
            self.style = 3  # Set this button to green
            self.parent_view.adetailer = "Faces"
        else:
            self.style = 1  # Set the button to default color
            self.parent_view.adetailer = None
        await interaction.response.edit_message(view=self.parent_view)


class BatchButton(Button):
    def __init__(self, parent_view):
        super().__init__(label="Batch: 2", custom_id="batch", style=1)
        self.parent_view = parent_view
        self.batch_values = ['2', '4', '1']
        self.current_index = 0
        self.parent_view.batch_value = self.batch_values[self.current_index]

    async def callback(self, interaction):
        # Increment the index and loop back to 0 if necessary
        self.current_index = (self.current_index + 1) % 3
        self.label = f"Batch: {self.batch_values[self.current_index]}"
        self.parent_view.batch_value = self.batch_values[self.current_index]
        await interaction.response.edit_message(view=self.parent_view)


class PromptButton(Button):
    def __init__(self, label, prompt_index, parent_view):
        super().__init__(
            label=label,
            custom_id=f"prompt_{prompt_index}",
            emoji="🎨")
        self.parent_view = parent_view

    async def callback(self, interaction):
        try:
            await interaction.response.defer()

            # ratio user choice
            size_ratio = None
            if self.parent_view.selected_orientation == "Portrait" and self.parent_view.selected_size == "Medium":
                size_ratio = "Portrait_Medium (832x1216)"
            elif self.parent_view.selected_orientation == "Portrait" and self.parent_view.selected_size == "Big":
                size_ratio = "Portrait_Big (1024x1536)"
            elif self.parent_view.selected_orientation == "Landscape" and self.parent_view.selected_size == "Medium":
                size_ratio = "Landscape_Medium (1216x832)"
            elif self.parent_view.selected_orientation == "Landscape" and self.parent_view.selected_size == "Big":
                size_ratio =  "Landscape_Big (1536x1024)"

            # model user choice
            model_choice = self.parent_view.selected_model

            # adetailer user choice
            adetailer_choice = self.parent_view.adetailer

            # batch user choice
            batch_choice = str(self.parent_view.batch_value)

            prompt_index = int(self.custom_id.split("_")[1])
            prompt = self.parent_view.prompts[prompt_index]
            self.parent_view.ctx.called_from_button = True
            await StableCog.dream_handler(self.parent_view.ctx, prompt=prompt, size_ratio=size_ratio, 
                                          data_model=model_choice, adetailer=adetailer_choice,
                                          batch=batch_choice)
            await interaction.edit_original_response(view=self.parent_view)
        except Exception as e:
            print(f'The draw button broke: {str(e)}')
            self.disabled = True
            await interaction.response.edit_message(view=self.parent_view)
            await interaction.followup.send("I may have been restarted. This button no longer works.", ephemeral=True)


class RerollButton(Button):
    def __init__(self, parent_view):
        super().__init__(
            label="Reroll",
            custom_id="reroll",
            emoji="🔁")
        self.parent_view = parent_view

    async def callback(self, interaction):
        try:
            #print("Entering RerollButton callback")
            await interaction.response.defer()
            self.parent_view.ctx.called_from_reroll = True
            await self.parent_view.generate_cog.generate_handler(
                self.parent_view.ctx, 
                prompt=self.parent_view.prompt,
                num_prompts=self.parent_view.num_prompts,
                max_length=self.parent_view.max_length,
                temperature= self.parent_view.temperature,
                top_k = self.parent_view.top_k,
                repetition_penalty = self.parent_view.repetition_penalty
            )
            await interaction.edit_original_response(view=self.parent_view)
        except discord.InteractionResponded:
            pass
        except Exception as e:
            print(f'Reroll button broke: {str(e)}')
            self.disabled = True
            await interaction.response.edit_message(view=self.parent_view)
            await interaction.followup.send("I may have been restarted. This button no longer works.", ephemeral=True)


class DeleteButton(Button):
    def __init__(self, parent_view):
        super().__init__(
            label="Delete",
            custom_id="delete",
            emoji="❌")
        self.parent_view = parent_view

    async def callback(self, interaction):
        try:
            await interaction.message.delete()
        except Exception as e:
            print(f'The delete button broke: {str(e)}')
            self.disabled = True
            await interaction.response.edit_message(view=self.parent_view)
            await interaction.followup.send("I may have been restarted. This button no longer works.", ephemeral=True)


class GenerateView(View):
    def __init__(self, prompts, generate_cog, ctx, message, prompt, num_prompts, max_length, temperature, top_k, repetition_penalty):
        super().__init__(timeout=None)
        self.generate_cog = generate_cog
        self.ctx = ctx
        self.prompts = prompts
        self.message = message
        self.prompt = prompt
        self.num_prompts = num_prompts
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty 
        for i, prompt in enumerate(prompts):
            button = PromptButton(label=f"Draw {i+1}", prompt_index=i, parent_view=self)
            self.add_item(button)
        self.add_item(RerollButton(parent_view=self))
        self.add_item(DeleteButton(parent_view=self))
        self.add_item(PortraitButton(parent_view=self))
        self.add_item(LandscapeButton(parent_view=self))
        self.add_item(NormalButton(parent_view=self))
        self.add_item(BigButton(parent_view=self))
        self.add_item(ChromaButton(parent_view=self))
        self.add_item(YumeButton(parent_view=self))
        self.add_item(FaceDetailerButton(parent_view=self))
        self.add_item(BatchButton(parent_view=self))

        # Attributes to store the selected orientation and size
        self.selected_orientation = "Portrait"
        self.selected_size = "Medium"
        self.selected_model = "ZavyChromaXL"
        self.adetailer = None
        self.batch_value = 2

    async def interaction_check(self, interaction):
        return True


class GenerateCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.model_path = "core/DistilGPT2-Stable-Diffusion-V2/"
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=75, temperature=0.9, top_k=8, repetition_penalty=1.2)

    @commands.Cog.listener()
    async def on_ready(self):
        self.bot.add_view(GenerateView([], None, None, None, "", 1, 75, 0.9, 8, 1.2))

    @commands.slash_command(name='generate', description='Generates a prompt from text', guild_only=True)
    @option(
        'prompt',
        str,
        description='Your text to produce the prompt.',
        required=True,
    )
    @option(
        'num_prompts',
        int,
        description='The number of prompts to produce. (1-5) Default: 1',
        required=False,
    )
    @option(
        'max_length',
        int,
        description='The max length for the generated prompts. (15-150) Default: 75',
        required=False,
    )
    @option(
        'temperature',
        float,
        description='Higher temp will produce more diverse results, but with a risk of less coherent text. Default: 0.9',
        required=False,
    )
    @option(
        'top_k',
        int,
        description='The number of tokens to sample from at each step. Default: 8',
        required=False,
    )
    @option(
        'repetition_penalty',
        float,
        description='The penalty value for each repetition of a token. Default: 1.2',
        required=False,
    )
    async def generate_handler(self, ctx: discord.ApplicationContext, *,
                            prompt: str,
                            num_prompts: Optional[int]=1,
                            max_length: Optional[int]=75,
                            temperature: Optional[float]=0.9,
                            top_k: Optional[int]=8,
                            repetition_penalty: Optional[float]=1.2):

        called_from_reroll = getattr(ctx, 'called_from_reroll', False)
        current_prompt = 0

        print(f"/Generate request -- {ctx.author.name}#{ctx.author.discriminator} -- {num_prompts} prompt(s) of {max_length} tokens. Text: {prompt}")

        # sanity check
        if not prompt or prompt.isspace():
            await ctx.send_followup("The prompt cannot be empty or contain only whitespace.")
            return

        if not 1 <= num_prompts <= 5:
            await ctx.send_followup("The number of prompts must be between 1 and 5.")
            return

        if not 15 <= max_length <= 150:
            await ctx.send_followup("The maximum length must be between 15 and 150.")
            return

        if temperature == 0:
            await ctx.send_followup("The temperature must not be zero.")
            return

        if top_k == 0:
            await ctx.send_followup("The top_k value must not be zero.")
            return

        if repetition_penalty == 0:
            await ctx.send_followup("The repetition penalty must not be zero.")
            return
        
        default_values = {
            'num_prompts': 1,
            'max_length': 75,
            'temperature': 0.9,
            'top_k': 8,
            'repetition_penalty': 1.2
        }

        current_values = {
            'num_prompts': num_prompts,
            'max_length': max_length,
            'temperature': temperature,
            'top_k': top_k,
            'repetition_penalty': repetition_penalty
        }

        key_mapping = {
            'num_prompts': 'Number of Prompts',
            'max_length': 'Max Length',
            'temperature': 'Temperature',
            'top_k': 'Top K',
            'repetition_penalty': 'Repetition Penalty'
        }

        modified_args = [f"{key_mapping[key]}: ``{value}``" for key, value in current_values.items() if value != default_values[key]]
        if modified_args:
            args_message = " - ".join(modified_args)
            response_message = f"<@{ctx.author.id}>, {settings.messages_prompt()}\nQueue: ``{len(queuehandler.GlobalQueue.generate_queue)}`` - Your text: ``{prompt}``\n{args_message}"
        else:
            response_message = f"<@{ctx.author.id}>, {settings.messages_prompt()}\nQueue: ``{len(queuehandler.GlobalQueue.generate_queue)}`` - Your text: ``{prompt}``"

        # set up the queue
        if queuehandler.GlobalQueue.generate_thread.is_alive():
            queuehandler.GlobalQueue.generate_queue.append(queuehandler.GenerateObject(self, ctx, prompt, num_prompts, max_length, temperature, top_k, repetition_penalty, current_prompt))
        else:
            await queuehandler.process_generate(self, queuehandler.GenerateObject(self, ctx, prompt, num_prompts, max_length, temperature, top_k, repetition_penalty, current_prompt))
        
        if called_from_reroll:
            await ctx.channel.send(response_message)
        else:
            await ctx.send_response(response_message)

    def post(self, event_loop: AbstractEventLoop, post_queue_object: queuehandler.PostObject):
        event_loop.create_task(
            post_queue_object.ctx.channel.send(
                content=post_queue_object.content,
                embed=post_queue_object.embed,
                view=None
            )
        )
        if queuehandler.GlobalQueue.post_queue:
            self.post(event_loop, queuehandler.GlobalQueue.post_queue.pop(0))

    def dream(self, event_loop: AbstractEventLoop, queue_object: queuehandler.GenerateObject, num_prompts: int, max_length: int, temperature: float, top_k: int, repetition_penalty: float):

        # start the progression message task
        event_loop.create_task(GlobalQueue.update_progress_message_generate(self, queue_object, num_prompts))

        try:
            # generate the text
            prompts = []
            for i in range(num_prompts):
                res = self.pipe(
                    queue_object.prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty
                    )
                generated_text = res[0]['generated_text']
                prompts.append(generated_text)

                # Inform the progress function about the current prompt number
                queue_object.current_prompt = i + 2

                # update the leaderboard
                LeaderboardCog.update_leaderboard(queue_object.ctx.author.id, str(queue_object.ctx.author), "Generate_Count")

            # progression flag, job done
            queue_object.is_done = True

            # Schedule the task to create the view and send the message
            event_loop.create_task(self.send_with_view(prompts, queue_object.ctx, queue_object.prompt, num_prompts, max_length, temperature, top_k, repetition_penalty))

        except Exception as e:
            embed = discord.Embed(title='Generation failed', description=f'{e}\n{traceback.print_exc()}', color=0x00ff00)
            event_loop.create_task(queue_object.ctx.channel.send(embed=embed))

        # check each queue for any remaining tasks
        if queuehandler.GlobalQueue.generate_queue:
            event_loop.create_task(queuehandler.process_generate(self, queuehandler.GlobalQueue.generate_queue.pop(0)))

    async def send_with_view(self, prompts, ctx, prompt, num_prompts, max_length, temperature, top_k, repetition_penalty):

        # create embed
        title = "What about this as Prompt?!" if len(prompts) == 1 else "What about these as Prompts?!"
        numbered_prompts = [f"**Prompt {i+1}:**\n{prompt}" for i, prompt in enumerate(prompts)]
        embed = discord.Embed(title=title, description="\n\n".join(numbered_prompts), color=0x00ff00)

        # post to discord        
        message = await ctx.send(content=f'<@{ctx.author.id}>', embed=embed)

        # create view
        view = GenerateView(prompts, self, ctx, message, prompt, num_prompts, max_length, temperature, top_k, repetition_penalty)

        # Update the message with the view
        await message.edit(view=view)

def setup(bot):
    bot.add_cog(GenerateCog(bot))
