import aiohttp
import asyncio
import discord
import re
import time
from discord.ui import View, Button
from threading import Thread

from PIL import Image
import io
import base64
import contextlib

from core import settings


# the queue object for txt2image and img2img
class DrawObject:
    def __init__(self, cog, ctx, simple_prompt, prompt, negative_prompt, data_model, steps, width, height,
                 guidance_scale, sampler, seed, strength, init_image, batch, styles, highres_fix,
                 clip_skip, extra_net, epoch_time, adetailer, scheduler, distilled_cfg_scale, view):# poseref, ipadapter
        self.cog = cog
        self.ctx = ctx
        self.simple_prompt = simple_prompt
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.data_model = data_model
        self.steps = steps
        self.width = width
        self.height = height
        self.guidance_scale = guidance_scale
        self.distilled_cfg_scale = distilled_cfg_scale
        self.sampler = sampler
        self.scheduler = scheduler
        self.seed = seed
        self.strength = strength
        self.init_image = init_image
        self.batch = batch
        self.styles = styles
        self.adetailer = adetailer
        #self.poseref = poseref
        #self.ipadapter = ipadapter
        self.highres_fix = highres_fix
        self.clip_skip = clip_skip
        self.extra_net = extra_net
        self.epoch_time = epoch_time
        self.view = view
        self.user_id = ctx.author.id
        self.is_done = False


# the queue object for Deforum command
class DeforumObject:
    def __init__(self, cog, ctx, deforum_settings, view, job_id=None):
        self.cog = cog
        self.ctx = ctx
        self.deforum_settings = deforum_settings
        self.prompt = deforum_settings["prompts"]
        self.view = view
        self.user_id = ctx.author.id
        self.data_model = ""
        self.job_id = job_id
        self.is_done = False


# the queue object for extras - upscale
class UpscaleObject:
    def __init__(self, cog, ctx, resize, init_image, upscaler_1, upscaler_2, upscaler_2_strength, gfpgan, codeformer,
                 upscale_first, view):
        self.cog = cog
        self.ctx = ctx
        self.resize = resize
        self.init_image = init_image
        self.upscaler_1 = upscaler_1
        self.upscaler_2 = upscaler_2
        self.upscaler_2_strength = upscaler_2_strength
        self.gfpgan = gfpgan
        self.codeformer = codeformer
        self.upscale_first = upscale_first
        self.view = view


# the queue object for identify (interrogate)
class IdentifyObject:
    def __init__(self, cog, ctx, init_image, phrasing, view):
        self.cog = cog
        self.ctx = ctx
        self.init_image = init_image
        self.phrasing = phrasing
        self.view = view


# the queue object for generate
class GenerateObject:
    def __init__(self, cog, ctx, prompt, num_prompts, max_length, temperature, top_k, repetition_penalty, current_prompt, model):
        self.cog = cog
        self.ctx = ctx
        self.prompt = prompt
        self.num_prompts = num_prompts
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.current_prompt = current_prompt
        self.model = model
        self.is_done = False


# the queue object for posting to Discord
class PostObject:
    def __init__(self, cog, ctx, content, file, embed, view):
        self.cog = cog
        self.ctx = ctx
        self.content = content
        self.file = file
        self.embed = embed
        self.view = view

# view that holds the interrupt button for progress
class ProgressView(View):
    def __init__(self, user_id):
        super().__init__(timeout=None)
        self.user_id = user_id

    async def user_is_authorized(self, interaction):
        if str(interaction.user.id) == str(self.user_id):
            return True
        
        moderator_role_name = "Moderator"
        return any(role.name == moderator_role_name for role in interaction.user.roles)

    @discord.ui.button(
        label="Interrupt Job", 
        custom_id="button_interrupt",
        emoji="❌")
    async def button_interrupt(self, button, interaction):
        try:
            if not await self.user_is_authorized(interaction):
                await interaction.response.send_message("You are not authorized to interrupt this job.", ephemeral=True)
                return
            s = settings.authenticate_user()
            s.post(url=f'{settings.global_var.url}/sdapi/v1/interrupt')
            await interaction.response.edit_message(view=self)
        except Exception as e:
            await interaction.response.send_message("An error occurred: " + str(e), ephemeral=True)

    @discord.ui.button(
        label="Skip Image", 
        custom_id="button_skip",
        emoji="➡️")
    async def button_skip(self, button, interaction):
        try:
            if not await self.user_is_authorized(interaction):
                await interaction.response.send_message("You are not authorized to skip this job.", ephemeral=True)
                return
            s = settings.authenticate_user()
            s.post(url=f'{settings.global_var.url}/sdapi/v1/skip')
            await interaction.response.edit_message(view=self)
        except Exception as e:
            await interaction.response.send_message("An error occurred: " + str(e), ephemeral=True)

    @discord.ui.button(
        label="Cancel All", 
        custom_id="button_cancel_all",
        emoji="🚫")
    async def button_cancel_all(self, button, interaction):
        if not await self.user_is_authorized(interaction):
            await interaction.response.send_message("You are not authorized to cancel jobs.", ephemeral=True)
            return

        # delete Draws jobs from user
        GlobalQueue.queue = [
            job for job in GlobalQueue.queue
            if not (isinstance(job, DrawObject) and job.user_id == self.user_id)
        ]

        await interaction.response.send_message("All pending draw jobs from you have been cancelled.", ephemeral=True)


# any command that needs to wait on processing should use the dream thread
class GlobalQueue:
    # progression lock and prioritys
    progress_lock = asyncio.Lock()

    dream_thread = Thread()
    post_event_loop = asyncio.get_event_loop()
    queue: list[DrawObject | UpscaleObject | IdentifyObject| DeforumObject] = []

    # new generate Queue
    generate_queue: list[GenerateObject] = []
    generate_thread = Thread()

    post_thread = Thread()
    event_loop = asyncio.get_event_loop()
    post_queue: list[PostObject] = []

    def get_queue_sizes():
        output = {}
        # Ajout d'un espace réservé pour "Queue Sizes" pour qu'il agisse comme un titre.
        output["General Queue Size"] = len(GlobalQueue.queue)
        output["Generate Queue Size"] = len(GlobalQueue.generate_queue)

        # Mapping des types d'objets à leurs noms d'affichage
        display_names = {
            "DrawObject": "Draw",
            "UpscaleObject": "Upscale",
            "IdentifyObject": "Identify",
            "DeforumObject": "Deforum"
        }

        if GlobalQueue.queue:
            general_queue_info = []
            for index, item in enumerate(GlobalQueue.queue[:5], start=1):
                item_info = f"\n{index}. {display_names.get(item.__class__.__name__, item.__class__.__name__)}"
                if isinstance(item, DrawObject):
                    item_info += f" - Prompt: {item.prompt[:100] + '...' if len(item.prompt) > 100 else item.prompt}"
                elif isinstance(item, DeforumObject):
                    deforum_prompt = str(item.prompt)
                    item_info += f" - Prompt: {deforum_prompt[:100] + '...' if len(item.prompt) > 100 else item.prompt}"
                general_queue_info.append(item_info)
            output["\n**General Queue next items**"] = "".join(general_queue_info)

        if GlobalQueue.generate_queue:
            generate_queue_info = []
            for index, item in enumerate(GlobalQueue.generate_queue[:5], start=1):
                item_info = f"\n{index}. **Generate** ({item.num_prompts} prompts)"
                generate_queue_info.append(item_info)
            output["\n**Generate Queue next items**"] = "".join(generate_queue_info)

        return output

    @staticmethod
    def create_progress_bar(progress, total_batches=1, length=20, empty_char='□', filled_char='■', batch_char='▨', filled_batch_char='▣'):
        filled_length = int(length * progress // 100)
     
        # mark batches
        batch_marker_positions = set(int(length * i // total_batches) for i in range(1, total_batches))
        bar = []
        for i in range(length):
            # not more than available slots
            if i < filled_length:
                if i in batch_marker_positions:
                    bar.append(filled_batch_char)
                else:
                    bar.append(filled_char)
            else:
                if i in batch_marker_positions:
                    bar.append(batch_char)
                else:
                    bar.append(empty_char)
        return f"`[{''.join(bar)}]`"

    @staticmethod
    async def handle_rate_limit(exception: discord.HTTPException, default_sleep=2):
        if exception.status == 429:
            #retry_after = float(exception.headers.get("Retry-After", default_sleep))
            #await asyncio.sleep(retry_after)
            ...
            return True
        return False

    @staticmethod
    async def update_progress_message(queue_object):
        async with GlobalQueue.progress_lock:
            ctx = getattr(queue_object, "ctx", None)
            #prompt = getattr(queue_object, "prompt", None)

            try:
                if isinstance(queue_object, DeforumObject) and "prompts" in queue_object.deforum_settings:
                    prompt_value = queue_object.deforum_settings["prompts"]
                    prompt = str(prompt_value)
                else:
                    prompt = getattr(queue_object, "prompt", "No prompt")
            except AttributeError:
                prompt = "No prompt"
            
            if isinstance(prompt, dict):
                try:
                    short_prompt = str(list(prompt.values())[0])[:125] + "..."
                except Exception:
                    short_prompt = "No prompt"
            else:
                short_prompt = prompt[:125] + "..." if len(prompt) > 125 else prompt

            # check for an existing progression message, if yes delete the previous one
            async for old_msg in ctx.channel.history(limit=25):
                if old_msg.embeds:
                    if old_msg.embeds[0].title == "        ──── Running Job Progression ────":
                        await old_msg.delete()

            # send first message to discord, Initialization
            embed = discord.Embed(title="Initialization...", color=discord.Color.blue())
            view = ProgressView(queue_object.user_id)

            progress_msg = await ctx.send(embed=embed, view=view)

            # Progress loop
            while not queue_object.is_done:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://127.0.0.1:7860/sdapi/v1/progress?skip_current_image=false") as response:

                        # Handle potential rate limit by Discord
                        try:
                            await progress_msg.edit(embed=embed, view=view)
                        except discord.HTTPException as e:
                            if await GlobalQueue.handle_rate_limit(e):
                                continue

                        data = await response.json()
                        try:
                            progress = round(data["progress"] * 100)
                            job = data['state']['job']

                            # Parsing the 'job' string to get the current and total number of batches
                            match = re.search(r'Batch (\d+) out of (\d+)', job)
                            if match:
                                current_batch, total_batches = map(int, match.groups())
                            else:
                                current_batch, total_batches = 1, 1

                            progress_bar = GlobalQueue.create_progress_bar(progress, total_batches=total_batches)
                            eta_relative = round(data["eta_relative"])
                            if prompt:
                                short_prompt = prompt[:125] + "..." if len(prompt) > 125 else prompt
                            else:
                                short_prompt = "No prompt"
                            sampling_step = data['state']['sampling_step']
                            sampling_steps = data['state']['sampling_steps']
                            queue_size = len(GlobalQueue.queue)

                            image_file = None
                            if data["current_image"] and data["current_image"].strip():
                                try:
                                    image_data = base64.b64decode(data["current_image"])
                                    if not image_data:
                                        #print("Error: image_data is empty after base64 decoding.")
                                        image_file = None
                                        await asyncio.sleep(2)
                                    else:
                                        image = Image.open(io.BytesIO(image_data))
                                        #print("Preview image successfully decoded and opened (format:", image.format, ")")

                                        # Always convert to RGB to support PNGs with alpha channel
                                        if image.mode in ("RGBA", "P"):
                                            image = image.convert("RGB")

                                        new_width = int(image.width * 1.5)
                                        new_height = int(image.height * 1.5)
                                        image = image.resize((new_width, new_height), Image.LANCZOS)

                                        with contextlib.ExitStack() as stack:
                                            buffer = stack.enter_context(io.BytesIO())
                                            image.save(buffer, 'JPEG')
                                            buffer.seek(0)
                                            # ----------- SEED FIX -----------
                                            if hasattr(queue_object, "seed"):
                                                filename = f"{queue_object.seed}.jpeg"
                                            elif hasattr(queue_object, "deforum_settings") and "seed" in queue_object.deforum_settings:
                                                filename = f"{queue_object.deforum_settings['seed']}.jpeg"
                                            else:
                                                filename = "preview.jpeg"
                                            image_file = discord.File(fp=buffer, filename=filename)
                                        #print("Preview image prepared for Discord")
                                except Exception as e:
                                    #print("Error while processing preview image:", repr(e))
                                    image_file = None
                            else:
                                #print("No preview image found (current_image is empty)")
                                await asyncio.sleep(2)

                            # Adjust job output to the running task
                            if job == "scripts_txt2img":
                                job = "Batch 1 out of 1"
                            elif job.startswith("task"):
                                job = "Job running locally by the owner"
                            elif job == "(unknown)":
                                job = "Ultimate Upscale"
                            elif job == "scripts_img2img":
                                job = "Prepare img2img script"

                            # Check recent messages and ensure the progress message is at the bottom
                            latest_message = await ctx.channel.history(limit=1).flatten()
                            latest_message = latest_message[0] if latest_message else None
                            if latest_message and latest_message.id != progress_msg.id:
                                await progress_msg.delete()
                                progress_msg = await ctx.send(embed=embed, view=view)

                            # Message update with fields
                            embed = discord.Embed(
                                title="──── Running Job Progression ────",
                                color=discord.Color.random()
                            )

                            # Clear existing fields
                            embed.clear_fields()

                            # Add fields to the embed
                            embed.add_field(name="**Prompt**", value=short_prompt, inline=False)
                            embed.add_field(name="📊 Progress", value=f"{progress_bar} {progress}%", inline=False)
                            embed.add_field(name="⏳ Remaining", value=f"{eta_relative} sec", inline=True)
                            embed.add_field(name="🔍 Current Step", value=f"{sampling_step}/{sampling_steps} - {job}", inline=True)
                            embed.add_field(name="👥 Queued Jobs", value=str(queue_size), inline=True)

                            # Set the image in the embed
                            embed.set_image(url=f"attachment://{queue_object.seed}.jpeg")

                            # Edit the progress message
                            await progress_msg.edit(embed=embed, file=image_file, view=view)

                            # Wait to avoid being rate limited by Discord
                            if isinstance(queue_object, DrawObject):
                                await asyncio.sleep(2)
                            elif isinstance(queue_object, DeforumObject):
                                await asyncio.sleep(2)
                            else:
                                await asyncio.sleep(2)

                        except Exception as e:
                            pass

            # Done, delete the progress message
            await progress_msg.delete()


    def process_queue():
        def start(target_queue: list[DrawObject | UpscaleObject | IdentifyObject | GenerateObject | DeforumObject]):
            queue_object = target_queue.pop(0)
            queue_object.cog.dream(GlobalQueue.event_loop, queue_object)

        if GlobalQueue.queue:
            start(GlobalQueue.queue)


async def process_dream(self, queue_object: DrawObject | UpscaleObject | IdentifyObject | DeforumObject):
    GlobalQueue.dream_thread = Thread(target=self.dream, args=(GlobalQueue.event_loop, queue_object))
    GlobalQueue.dream_thread.start()

async def process_generate(generate_cog, queue_object: GenerateObject):
    GlobalQueue.generate_thread = Thread(target=generate_cog.dream, args=(
        GlobalQueue.event_loop, queue_object, queue_object.num_prompts, queue_object.max_length, 
        queue_object.temperature, queue_object.top_k, queue_object.repetition_penalty, queue_object.model))
    GlobalQueue.generate_thread.start()

def process_post(self, queue_object: PostObject):
    if GlobalQueue.post_thread.is_alive():
        GlobalQueue.post_queue.append(queue_object)
    else:
        GlobalQueue.post_thread = Thread(target=self.post, args=(GlobalQueue.post_event_loop, queue_object))
        GlobalQueue.post_thread.start()
