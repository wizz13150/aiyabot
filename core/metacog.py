import discord
import requests
import random
from PIL import Image
from io import BytesIO
from typing import Optional
from discord import option, ui, Button, Embed
from discord.ext import commands
from discord.ui import Button, View

from core import settings
from core.stablecog import StableCog


# Define a View for the metadata buttons
class MetaView(ui.View):
    def __init__(self, ctx, metadata_raw, prompt, negative_prompt, steps, sampler, cfg_scale, seed, size, clip_skip):
        super().__init__(timeout=None)
        
        self.ctx = ctx
        self.metadata_raw = metadata_raw
        self.prompt = prompt if prompt != "N/A" else None
        self.negative_prompt = negative_prompt if negative_prompt != "N/A" else None
        self.steps = int(steps) if steps != "N/A" else None
        self.sampler = sampler if sampler != "N/A" else None
        self.cfg_scale = cfg_scale if cfg_scale != "N/A" else None
        self.seed = int(seed) if seed != "N/A" else None
        self.size = size if size != "N/A" else None
        self.clip_skip = int(clip_skip) if clip_skip != "N/A" else None
        # self.data_model = data_model if data_model != "N/A" else None # Down until v1.6.0 webui is fixed to load old SD1.5 models

    @discord.ui.button(
        custom_id="button_draw_from_meta",
        emoji="🎨",
        label="Draw From Meta")
    async def draw_from_meta(self, button: ui.Button, interaction: discord.Interaction):
        try:
            await interaction.response.defer()
            
            # Construct dynamic arguments for StableCog.dream_handler
            dream_args = {}
        
            if self.prompt is not None:
                dream_args['prompt'] = self.prompt
            if self.negative_prompt is not None:
                dream_args['negative_prompt'] = self.negative_prompt
            if self.steps is not None:
                dream_args['steps'] = self.steps
            if self.size is not None:
                width, height = map(int, self.size.split("x"))
                dream_args['width'] = width
                dream_args['height'] = height
            if self.cfg_scale is not None:
                dream_args['guidance_scale'] = self.cfg_scale
            if self.sampler is not None:
                dream_args['sampler'] = self.sampler
            if self.seed is not None:
                dream_args['seed'] = self.seed
            if self.clip_skip is not None:
                dream_args['clip_skip'] = self.clip_skip
                
            # Model selection
            #if self.data_model is not None and self.data_model in settings.global_var.model_info:
            #    dream_args['data_model'] = self.data_model            
            
            # Using **dream_args unpacks the dictionary into keyword arguments for the function
            self.ctx.called_from_button = True
            await StableCog.dream_handler(self.ctx, **dream_args)
            await interaction.edit_original_response(view=self)
        except Exception as e:
            print(f'The Draw from Meta button broke: {str(e)}')
            self.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send("I may have been restarted. This button no longer works.", ephemeral=True)    
            
    # Define a button for copying generation data
    @discord.ui.button(
        custom_id="button_copy_generation_data_meta",
        emoji="📋",
        label="Copy Generation Datas")
    async def copy_generation_data(self, button: ui.Button, interaction: discord.Interaction):
        try:
            metadata_cleaned = self.metadata_raw.replace("\n", ", ")
         
            # Create the Embed and mention the user who triggered the interaction in the description
            embed = Embed(
                title="──── Generation Datas────", 
                description=f"```\nPrompt: {metadata_cleaned}\n```", 
                color=random.randint(0, 0xFFFFFF)
            )     
            await interaction.response.send_message(content=f"<@{interaction.user.id}>", embed=embed)
        except Exception as e:
            print(f'The Copy Generation Datas button broke: {str(e)}')
            self.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send("I may have been restarted. This button no longer works.", ephemeral=True)  

    # Define a button for deleting the metadata message
    @discord.ui.button(
        custom_id="delete_meta",
        emoji="❌",
        label="Delete")
    async def delete_meta(self, button: ui.Button, interaction: discord.Interaction):
        await interaction.message.delete()


# Define the MetaCog class
class MetaCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    commands.Cog.listener()
    async def on_ready(self):
        self.bot.add_view(MetaView(self))

    # Define a slash command for extracting metadata
    @commands.slash_command(name='meta', description='Extract Generation datas from an image', guild_only=True)
    @option(
        'init_image',
        discord.Attachment,
        description='The image to extract metadata from.',
        required=False,
    )
    @option(
        'init_url',
        str,
        description='The URL image to extract metadata from. This overrides init_image!',
        required=False,
    )
    async def meta_handler(self, ctx, init_image: Optional[discord.Attachment] = None, init_url: Optional[str] = None):

        print(f"/Meta request -- {ctx.author.name}#{ctx.author.discriminator} ... Image: {init_image if init_image else 'None'}, URL: {init_url if init_url else 'None'}")
        
        has_image = True
        if init_url:
            try:
                response = requests.get(init_url)
                image_data = BytesIO(response.content)
            except requests.RequestException:
                await ctx.respond("🚫 Couldn't fetch image from the URL.", ephemeral=True)
                return
        elif init_image:
            image_data = BytesIO()
            await init_image.save(image_data)
        else:
            await ctx.respond("🚫 No image provided.", ephemeral=True)
            return
        
        image_data.seek(0)
        image = Image.open(image_data)
        metadata = image.info.get('parameters', '')  # Assume the metadata you want is under the 'parameters' key

        if metadata:
            # Extract different parts as you specified
            prompt_end = metadata.find("Negative prompt:") if "Negative prompt:" in metadata else metadata.find("Steps:")
            prompt = metadata[:prompt_end].strip()

            if "Negative prompt:" in metadata and "Steps:" in metadata:
                negative_prompt_start = metadata.find("Negative prompt:") + len("Negative prompt:")
                negative_prompt_end = metadata.find("Steps:")
                negative_prompt = metadata[negative_prompt_start:negative_prompt_end].strip()
            else:
                negative_prompt = "N/A"

            if "Steps:" in metadata:
                steps_and_beyond = metadata.split("Steps:")[1].strip()
                steps = steps_and_beyond.split(", ")[0]  # First value after "Steps:"
            else:
                steps_and_beyond = ""
                steps = "N/A"

            # Process the remaining key-value pairs
            key_value_pairs = [seg.split(": ", 1) for seg in steps_and_beyond.split(", ") if ": " in seg]
            metadata_dict = {k: v for k, v in key_value_pairs}

            # Filter the keys you want
            keys_to_extract = ['Model', 'Seed', 'Steps', 'CFG scale', 'Sampler', 'Size', 'Clip skip', 'Model']
            extracted_metadata = {k: metadata_dict.get(k, 'N/A') for k in keys_to_extract}

            # Extract necessary information from extracted_metadata
            sampler = extracted_metadata['Sampler']
            cfg_scale = extracted_metadata['CFG scale']
            seed = extracted_metadata['Seed']
            size = extracted_metadata['Size']
            clip_skip = extracted_metadata['Clip skip']
            data_model = extracted_metadata['Model']  

            # Create the full Embed description from the extracted metadata
            full_extracted_metadata_str = f"🌟 **Model:** `{metadata_dict.get('Model', 'N/A')}`\n"
            full_extracted_metadata_str += f"🎨 **Prompt:**\n`{prompt}`\n🚫 **Negative Prompt:**\n`{negative_prompt}`\n"
            full_extracted_metadata_str += f"🌱 **Seed:** `{metadata_dict.get('Seed', 'N/A')}`\n🔢 **Steps:** `{steps}`\n🔍 **CFG scale:** `{metadata_dict.get('CFG scale', 'N/A')}`\n"
            for key in ['Sampler', 'Size', 'Clip skip']:
                full_extracted_metadata_str += f"🛠️ **{key}:** `{metadata_dict.get(key, 'N/A')}`\n"

            # Create a truncated version for Discord display
            display_extracted_metadata_str = full_extracted_metadata_str
            if len(display_extracted_metadata_str) > 1024:
                display_extracted_metadata_str = display_extracted_metadata_str[:1020] + "..."

            # Group the other keys and values into a single block
            other_metadata_str = "Model hash:" + steps_and_beyond.split("Model hash:")[1].strip()  # Start at "Model hash"
            other_metadata_str = other_metadata_str.replace(f"Steps: {steps}, ", "")
            other_metadata_str = other_metadata_str.rstrip(', ')  # Remove the last comma

            # Create an Embed with a random color
            embed = discord.Embed(title="──── Generation Datas ────", color=random.randint(0, 0xFFFFFF))
            embed.add_field(name='\u200b', value=display_extracted_metadata_str, inline=False)
            embed.add_field(name="🔍 Other Metadata 🔍", value=other_metadata_str, inline=False)

            # Rewind the BytesIO object to the beginning
            image_data.seek(0)
            file = discord.File(image_data, filename="preview.png")
            embed.set_thumbnail(url="attachment://preview.png")

            # Create an instance of MetaView and pass the raw metadata
            view = MetaView(ctx, metadata, prompt, negative_prompt, steps, sampler, cfg_scale, seed, size, clip_skip)

            # Add the view when you send the message
            await ctx.respond(content=f'<@{ctx.author.id}>', embed=embed, file=file, view=view)
        else:
            await ctx.respond("🚫 No metadata found. Please try again with a valid image! 📸", ephemeral=True)

def setup(bot):
    bot.add_cog(MetaCog(bot))
