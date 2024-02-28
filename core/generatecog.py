import discord
import traceback
from asyncio import AbstractEventLoop, get_event_loop, run_coroutine_threadsafe
from discord import option
from discord.ui import Button, View, Select
from discord.ext import commands
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from typing import Optional

from core import queuehandler
from core import settings
from core import settingscog
from core.queuehandler import GlobalQueue
from core.stablecog import StableCog
from core.leaderboardcog import LeaderboardCog


class RatioButton(Button):
    FORMATS = [
        "Portrait: 2:3 - 768x1280",
        "Landscape: 3:2 - 1280x768",
        "Fullscreen: 4:3 - 1152x896",
        "Widescreen: 16:9 - 1344x768",
        "Ultrawide: 21:9 - 1536x640",
        "Square: 1:1 - 1024x1024",
        "Tall: 9:16 - 768x1344"
    ]
    
    def __init__(self, parent_view):
        super().__init__(label=self.FORMATS[0], custom_id="ratio", emoji="🔄", style=1)
        self.parent_view = parent_view
        self.parent_view.selected_format = self.FORMATS[0]
        self.current_index = 0

    async def callback(self, interaction):
        self.current_index = (self.current_index + 1) % len(self.FORMATS)
        self.label = self.FORMATS[self.current_index]
        self.parent_view.selected_format = self.label
        self.parent_view.update_select_menus()
        await interaction.response.edit_message(view=self.parent_view)


class ADetailerButton(Button):
    def __init__(self, parent_view):
        super().__init__(label="ADetailer", custom_id="adetailer", emoji="🎭", style=1)
        self.parent_view = parent_view
        self.choices = ["None", "Faces", "Hands", "Faces+Hands"]
        self.current_choice_index = 0  # Default is 'None'

    async def callback(self, interaction):
        # Change the current choice
        self.current_choice_index = (self.current_choice_index + 1) % len(self.choices)
        current_choice = self.choices[self.current_choice_index]
        
        # Update the button's label and style based on the current choice
        if current_choice == "None":
            self.label = "ADetailer"
            self.style = 1  # Default color
        else:
            self.label = f"ADetailer {current_choice}"
            self.style = 3  # Green

        self.parent_view.adetailer = current_choice if current_choice != "None" else None
        self.parent_view.update_select_menus()
        await interaction.response.edit_message(view=self.parent_view)


class HighResButton(Button):
    def __init__(self, parent_view):
        super().__init__(label="HighResFix", custom_id="highres", emoji="🔍", style=1)
        self.parent_view = parent_view

    async def callback(self, interaction):
        if self.style == 1:  # Default color
            self.style = 3  # Set this button to green
            self.parent_view.hires = "4x_foolhardy_Remacri"
        else:
            self.style = 1  # Set the button to default color
            self.parent_view.hires = None
        self.parent_view.update_select_menus()
        await interaction.response.edit_message(view=self.parent_view)


class BatchButton(Button):
    def __init__(self, parent_view):
        super().__init__(label="Batch: 2", custom_id="batch", style=1)
        self.parent_view = parent_view
        self.batch_values = ['2', '4', '9', '1']
        self.current_index = 0
        self.parent_view.batch_value = self.batch_values[self.current_index]

    async def callback(self, interaction):
        # Increment the index and loop back to 0 if necessary
        self.current_index = (self.current_index + 1) % 4
        self.label = f"Batch: {self.batch_values[self.current_index]}"
        self.parent_view.batch_value = self.batch_values[self.current_index]
        self.parent_view.update_select_menus()
        await interaction.response.edit_message(view=self.parent_view)


class PromptButton(Button):
    def __init__(self, label, prompt_index, parent_view):
        super().__init__(
            label=label,
            custom_id=f"prompt_{prompt_index}",
            emoji="🎨")
        self.parent_view = parent_view
        self.prompt_index = prompt_index

    async def callback(self, interaction):
        try:
            await interaction.response.defer()
            await self.parent_view.handle_draw_prompt(interaction, self.parent_view.prompts[self.prompt_index], self.prompt_index)
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


class DrawAllButton(Button):
    def __init__(self, parent_view):
        super().__init__(label="Draw All", custom_id="draw_all", emoji="✨", style=1)
        self.parent_view = parent_view

    async def callback(self, interaction):
        try:
            await interaction.response.defer()

            # Récupérez toutes les prompts de la vue parente
            all_prompts = self.parent_view.prompts

            # Générez chaque prompt
            for i, prompt in enumerate(all_prompts):
                await self.parent_view.handle_draw_prompt(interaction, prompt, i)

            # Éditez la réponse originale avec la vue mise à jour
            await interaction.edit_original_response(view=self.parent_view)
        except Exception as e:
            print(f'Draw All button broke: {str(e)}')
            self.disabled = True
            await interaction.response.edit_message(view=self.parent_view)
            await interaction.followup.send("Une erreur s'est produite lors de la génération de toutes les prompts.", ephemeral=True)


class LorasSelect(Select):
    def __init__(self, loras_list, *args, **kwargs):
        options = [discord.SelectOption(label=lora) for lora in loras_list if lora is not None and lora != "None"]
        super().__init__(placeholder="Select Loras to apply (Multiplier 0.85)...", min_values=1, max_values=len(options), options=options, *args, **kwargs)
    
    async def callback(self, interaction: discord.Interaction):
        selected_loras = self.values
        self.view.loras_selections = selected_loras
        await interaction.response.defer()

    def update_selected_options(self, selected_loras):
        for option in self.options:
            option.default = option.value in selected_loras


class StylesSelect(Select):
    def __init__(self, styles_list, *args, **kwargs):
        limited_styles_list = styles_list[:25]
        super().__init__(*args, **kwargs)
        self.placeholder = "Select a Style to apply..."
        self.min_values = 1
        self.max_values = 1 # Fixed to 1 choice for now # min(len(limited_styles_list), 25)
        self.options = [
            discord.SelectOption(label=style) for style in limited_styles_list
        ]
    
    async def callback(self, interaction: discord.Interaction):
        selected_styles = self.values
        self.view.styles_selections = selected_styles
        await interaction.response.defer()

    def update_selected_options(self, selected_styles):
        for option in self.options:
            option.default = option.value in selected_styles

'''
class ModelsSelect(Select):
    def __init__(self, models_list, *args, **kwargs):
        limited_models_list = models_list[:25]
        super().__init__(*args, **kwargs)
        self.placeholder = "Select another Model to use..."
        self.min_values = 1
        self.max_values = 1
        self.options = [
            discord.SelectOption(label=model) for model in limited_models_list
        ]
    
    async def callback(self, interaction: discord.Interaction):
        selected_models = self.values
        self.view.models_selections = selected_models
        #print(f"Selected Model: {selected_models[0]}")
        await interaction.response.defer()
'''

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
        self.loras_selections = []
        #self.models_selections = []
        self.styles_selections = []

        for i, prompt in enumerate(prompts):
            button = PromptButton(label=f"Draw {i+1}", prompt_index=i, parent_view=self)
            self.add_item(button)

        self.add_item(RerollButton(parent_view=self))
        self.add_item(DeleteButton(parent_view=self))
        self.add_item(RatioButton(parent_view=self))
        self.add_item(ADetailerButton(parent_view=self))
        self.add_item(HighResButton(parent_view=self))
        self.add_item(BatchButton(parent_view=self))
        self.add_item(DrawAllButton(parent_view=self))

        # Create and add the Models dropdown menu
        settings_cog = settingscog.SettingsCog(self)
        #models_list = settings_cog.model_autocomplete()
        #self.models_select = ModelsSelect(models_list)
        #self.add_item(self.models_select)

        # Create and add the Lora dropdown menu
        settings_cog = settingscog.SettingsCog(self)
        loras_list = settings_cog.extra_net_autocomplete()
        self.loras_select = LorasSelect(loras_list)
        self.add_item(self.loras_select)

        # Create and add the Styles dropdown menu
        styles_list = settings_cog.style_autocomplete()
        self.styles_select = StylesSelect(styles_list)
        self.add_item(self.styles_select)

        self.selected_orientation = "Portrait: 2:3 - 768x1280"
        self.selected_model = "ZavyChromaXL"
        self.hires = None
        self.adetailer = None
        self.batch_value = 2

    async def interaction_check(self, interaction):
        return True

    def build_loras_tags(self):
            tags = []
            for lora in self.loras_selections:
                tag = f"<lora:{lora}:0.85>"
                tags.append(tag)
            return " ".join(tags)

    async def handle_draw_prompt(self, interaction, prompt, prompt_index):
        size_ratio = self.selected_format
        adetailer_choice = self.adetailer
        highres_choice = self.hires
        batch_choice = str(self.batch_value)
        loras_tags = self.build_loras_tags()
        prompt = f"{prompt} {loras_tags}"
        styles = ",".join(self.styles_selections) if self.styles_selections else None

        self.ctx.called_from_button = True
        await StableCog.dream_handler(self.ctx, prompt=prompt,
                                    styles=styles,
                                    size_ratio=size_ratio,
                                    adetailer=adetailer_choice,
                                    highres_fix=highres_choice,
                                    batch=batch_choice)

        self.update_select_menus()
        await interaction.edit_original_response(view=self)

    def update_select_menus(self):
        for item in self.children:
            if isinstance(item, LorasSelect):
                item.update_selected_options(self.loras_selections)
            elif isinstance(item, StylesSelect):
                item.update_selected_options(self.styles_selections)


class GenerateCog(commands.Cog):
    model_paths = {
            #"WizzGPTV2": "core/WizzGPT2-v2",
            #"Insomnia": "core/Insomnia",
            #"WizzGPT": "core/WizzGPT2",
            "DistilGPT2-V2": "core/DistilGPT2-Stable-Diffusion-V2",
            "MagicPrompt-SD": "core/MagicPrompt-SD/"#,
            #"Microsoft-Promptist": "core/Microsoft-Promptist",
            #"Daspartho-Prompt-extend": "core/Daspartho-Prompt-extend", 
            #"LexicArt": "core/LexicArt", 
            #"MajinAI": "core/MajinAI", 
            #"Kmewhort-SD-prompt-bolster": "core/Kmewhort-SD-prompt-bolster",
            #"Succinctly-Pompt-generator": "core/Succinctly-Pompt-generator"
        }
    model_choices = list(model_paths.keys())

    def __init__(self, bot):
        self.bot = bot
        self.models = {}
        self.tokenizers = {}
        for model_name, model_path in self.model_paths.items():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
        self.current_model = "WizzGPTV2"

    @commands.slash_command(name='generate', description='Generates a prompt from text', guild_only=True)
    @option(
        'model',
        str,
        description='Choose the model to use to extend your prompt.',
        required=False,
        choices=model_choices
    )
    @option(
        'prompt',
        str,
        description='Your text to produce the prompt.',
        required=True,
    )
    @option(
        'num_prompts',
        int,
        description='The number of prompts to produce. (1-7) Default: 5',
        required=False,
    )
    @option(
        'max_length',
        int,
        description='The max length for the generated prompts. (15-125) Default: 75',
        required=False,
    )
    @option(
        'temperature',
        float,
        description='Higher temp will produce more diverse results, but with a risk of less coherent text. Default: 1.1',
        required=False,
    )
    @option(
        'top_k',
        int,
        description='The number of tokens to sample from at each step. Default: 24',
        required=False,
    )
    @option(
        'repetition_penalty',
        float,
        description='The penalty value for each repetition of a token. Default: 1.35',
        required=False,
    )
    async def generate_handler(self, ctx: discord.ApplicationContext, *,
                           prompt: str,
                           num_prompts: Optional[int] = 5,
                           max_length: Optional[int] = 75,
                           temperature: Optional[float] = 1.1,
                           top_k: Optional[int] = 24,
                           repetition_penalty: Optional[float] = 1.35,
                           model: Optional[str] = "WizzGPTV2"):
        
        self.current_model = model
        tokenizer = self.tokenizers[self.current_model]
        eos_token_id = tokenizer.eos_token_id
        self.pipe = pipeline('text-generation', model=self.models[self.current_model], tokenizer=tokenizer, max_length=75, temperature=0.7, top_k=8, repetition_penalty=1.2, eos_token_id=eos_token_id)

        called_from_reroll = getattr(ctx, 'called_from_reroll', False)
        current_prompt = 0

        print(f"/Generate request -- {ctx.author.name}#{ctx.author.discriminator} -- {num_prompts} prompt(s) of {max_length} tokens. Text: {prompt}")

        # sanity check
        if not prompt or prompt.isspace():
            await ctx.respond("The prompt cannot be empty or contain only whitespace.")
            return

        if not 1 <= num_prompts <= 7:
            await ctx.respond("The number of prompts must be between 1 and 7.")
            return

        if not 15 <= max_length <= 125:
            await ctx.respond("The maximum length must be between 15 and 125.")
            return

        if temperature == 0:
            await ctx.respond("The temperature must not be zero.")
            return

        if top_k == 0:
            await ctx.respond("The top_k value must not be zero.")
            return

        if repetition_penalty == 0:
            await ctx.respond("The repetition penalty must not be zero.")
            return
        
        default_values = {
            'num_prompts': "dummy", # always mention the number
            'max_length': "dummy", # always mention the length
            'temperature': 1.1,
            'top_k': 24,
            'repetition_penalty': 1.35,
            'model': "dummy" # always mention the model
        }

        current_values = {
            'num_prompts': num_prompts,
            'max_length': max_length,
            'temperature': temperature,
            'top_k': top_k,
            'repetition_penalty': repetition_penalty,
            'model': model
        }

        key_mapping = {
            'num_prompts': 'Number of Prompts',
            'max_length': 'Max Length',
            'temperature': 'Temperature',
            'top_k': 'Top K',
            'repetition_penalty': 'Repetition Penalty',
            'model': 'Model'
        }

        modified_args = [f"{key_mapping[key]}: ``{value}``" for key, value in current_values.items() if value != default_values[key]]
        if modified_args:
            args_message = " - ".join(modified_args)
            response_message = f"<@{ctx.author.id}>, {settings.messages_prompt()}\nQueue: ``{len(queuehandler.GlobalQueue.generate_queue)}`` - Your text: ``{prompt}``\n{args_message}"
        else:
            response_message = f"<@{ctx.author.id}>, {settings.messages_prompt()}\nQueue: ``{len(queuehandler.GlobalQueue.generate_queue)}`` - Your text: ``{prompt}``"

        # set up the queue
        if queuehandler.GlobalQueue.generate_thread.is_alive():
            queuehandler.GlobalQueue.generate_queue.append(queuehandler.GenerateObject(self, ctx, prompt, num_prompts, max_length, temperature, top_k, repetition_penalty, current_prompt, model))
        else:
            await queuehandler.process_generate(self, queuehandler.GenerateObject(self, ctx, prompt, num_prompts, max_length, temperature, top_k, repetition_penalty, current_prompt, model))
        
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

    def dream(self, event_loop: AbstractEventLoop, queue_object: queuehandler.GenerateObject, num_prompts: int, max_length: int, temperature: float, top_k: int, repetition_penalty: float, model: str):
        try:
            # Choisissez le modèle et le tokenizer en fonction de l'argument 'model'
            selected_model = self.models[model]
            selected_tokenizer = self.tokenizers[model]
            selected_eos_token_id = selected_tokenizer.eos_token_id

            # Configurez le pipeline avec le modèle et le tokenizer sélectionnés
            pipe = pipeline('text-generation', model=selected_model, tokenizer=selected_tokenizer, max_length=max_length, temperature=temperature, top_k=top_k, repetition_penalty=repetition_penalty, eos_token_id=selected_eos_token_id)

            # Générez le texte
            prompts = []
            for i in range(num_prompts):
                res = pipe(
                    queue_object.prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=selected_eos_token_id
                    )
                generated_text = res[0]['generated_text']
                prompts.append(generated_text)

                # Mise à jour du classement
                LeaderboardCog.update_leaderboard(queue_object.ctx.author.id, str(queue_object.ctx.author), "Generate_Count")

            # Planifiez la tâche pour créer la vue et envoyer le message
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
