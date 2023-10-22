import aiohttp
import asyncio
import discord
import re
from threading import Thread


# the queue object for txt2image and img2img
class DrawObject:
    def __init__(self, cog, ctx, simple_prompt, prompt, negative_prompt, data_model, steps, width, height,
                 guidance_scale, sampler, seed, strength, init_image, batch, styles, highres_fix,
                 clip_skip, extra_net, epoch_time, adetailer, poseref, view):
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
        self.sampler = sampler
        self.seed = seed
        self.strength = strength
        self.init_image = init_image
        self.batch = batch
        self.styles = styles
        self.adetailer = adetailer
        self.poseref = poseref
        self.highres_fix = highres_fix
        self.clip_skip = clip_skip
        self.extra_net = extra_net
        self.epoch_time = epoch_time
        self.view = view
        self.is_done = False


# the queue object for Deforum command
class DeforumObject:
    def __init__(self, cog, ctx, deforum_settings, view):
        self.cog = cog
        self.ctx = ctx
        self.deforum_settings = deforum_settings
        self.prompt = deforum_settings["prompts"]
        self.view = view
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
    def __init__(self, cog, ctx, prompt, num_prompts, max_length, temperature, top_k, repetition_penalty, current_prompt):
        self.cog = cog
        self.ctx = ctx
        self.prompt = prompt
        self.num_prompts = num_prompts
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.current_prompt = current_prompt
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


# any command that needs to wait on processing should use the dream thread
class GlobalQueue:
    # progression lock and prioritys
    progress_lock = asyncio.Lock()
    priority_flag = asyncio.Event()

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
    async def handle_rate_limit(response, default_sleep=2):
        """
        Handle the rate limit by checking the response's status and headers.
        Sleeps for the duration specified by the Retry-After header if rate-limited.
        """
        if response.status == 429:
            retry_after = float(response.headers.get("Retry-After", default_sleep))
            await asyncio.sleep(retry_after)
            return True
        return False

    @staticmethod
    async def update_progress_message(queue_object):
        async with GlobalQueue.progress_lock:
            # priority flag
            GlobalQueue.priority_flag.set()

            ctx = getattr(queue_object, "ctx", None)
            prompt = getattr(queue_object, "prompt", None)

            try:
                if "prompts" in queue_object.deforum_settings:
                    prompt_value = queue_object.deforum_settings["prompts"]
                    prompt = str(prompt_value)
                else:
                    print("[DEBUG] 'Prompts' not found in deforum_settings.")
            except AttributeError as e:
                ...

            # check for an existing progression message, if yes delete the previous one
            async for old_msg in ctx.channel.history(limit=25):
                if old_msg.embeds:
                    if old_msg.embeds[0].title == "──── Running Job Progression ────":
                        await old_msg.delete()

            # send first message to discord, Initialization
            embed = discord.Embed(title="Initialization...", color=discord.Color.blue())
            progress_msg = await ctx.send(embed=embed)

            # progress loop
            while not queue_object.is_done:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://127.0.0.1:7860/sdapi/v1/progress?skip_current_image=false") as response:
                        
                        # Handle potential rate limit by Discord
                        if await GlobalQueue.handle_rate_limit(response):
                            continue

                        data = await response.json()
                        try:
                            progress = round(data["progress"] * 100)
                            job = data['state']['job']

                            # parsing the 'job' string to get the current and total number of batches
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

                            # adjust job output to the running task
                            if job == "scripts_txt2img":
                                job = "Batch 1 out of 1"
                            elif job.startswith("task"):
                                job = "Job running locally by the owner"

                            # check recent messages and Spam the bottom, like pinned
                            latest_message = await ctx.channel.history(limit=1).flatten()
                            latest_message = latest_message[0] if latest_message else None
                            if latest_message and latest_message.id != progress_msg.id:
                                await progress_msg.delete()
                                progress_msg = await ctx.send(embed=embed)

                            # message update
                            embed = discord.Embed(title=f"──── Running Job Progression ────", 
                                                description=f"**Prompt**: {short_prompt}\n📊 {progress_bar} {progress}%\n⏳ **Remaining**: {eta_relative} seconds\n🔍 **Current Step**: {sampling_step}/{sampling_steps}  -  {job}\n👥 **Queued Jobs**: {queue_size}", 
                                                color=discord.Color.random())
                            await progress_msg.edit(embed=embed)

                            # wait 2 or 5 to not be rate limited by discord
                            if isinstance(queue_object, DrawObject):
                                await asyncio.sleep(2)
                            elif isinstance(queue_object, DeforumObject):
                                await asyncio.sleep(5)
                            else:
                                await asyncio.sleep(2)

                        except Exception as e:
                            print(f"[ERROR] Error regarding the server response: {e}")

            # done, delete, clear priority flag
            await progress_msg.delete()
            GlobalQueue.priority_flag.clear()

    async def update_progress_message_generate(instance, queue_object, num_prompts):
        # wait for the priority flag to end
        while GlobalQueue.priority_flag.is_set():
            await asyncio.sleep(1)

        # start only if no lock AND no priority flag. double is better, they said
        async with GlobalQueue.progress_lock:
            if not GlobalQueue.priority_flag.is_set():
                ctx = queue_object.ctx
                prompt = queue_object.prompt

                # check for an existing progression message, if yes delete the previous one
                async for old_msg in ctx.channel.history(limit=25):
                    if old_msg.embeds:
                        if old_msg.embeds[0].title == "──── Running Job Progression ────":
                            await old_msg.delete()

                # send first message to discord, Initialization
                embed = discord.Embed(title="Initialization...", color=discord.Color.blue())
                progress_msg = await ctx.send(embed=embed)

                # update progress message based on the current prompt being generated
                while not queue_object.is_done:
                    queue_size = len(GlobalQueue.generate_queue)
                    description = f"Generating {num_prompts} {'prompt' if num_prompts == 1 else 'prompts'}!"
                    description += f"\n**Prompt**: {prompt}\n🔍**Current Prompt**: {queue_object.current_prompt} of {num_prompts}\n👥 **Queued Jobs**: {queue_size}"
                    embed = discord.Embed(title=f"──── Running Job Progression ────", description=description, color=discord.Color.random())
                    
                    # handle potential rate limit by Discord for editing the message
                    if await GlobalQueue.handle_rate_limit(await progress_msg.edit(embed=embed)):
                        continue

                    # check if the message has been moved in the chat and move it down if needed
                    latest_message = await ctx.channel.history(limit=1).flatten()
                    latest_message = latest_message[0] if latest_message else None

                    if latest_message and latest_message.id != progress_msg.id:
                        await progress_msg.delete()
                        progress_msg = await ctx.send(embed=embed)
                    
                    await asyncio.sleep(2)

        # done, delete
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

async def process_generate(self, queue_object: GenerateObject):
    GlobalQueue.generate_thread = Thread(target=self.dream, args=(GlobalQueue.event_loop, queue_object, queue_object.num_prompts, queue_object.max_length, queue_object.temperature, queue_object.top_k, queue_object.repetition_penalty))
    GlobalQueue.generate_thread.start()

def process_post(self, queue_object: PostObject):
    if GlobalQueue.post_thread.is_alive():
        GlobalQueue.post_queue.append(queue_object)
    else:
        GlobalQueue.post_thread = Thread(target=self.post, args=(GlobalQueue.post_event_loop, queue_object))
        GlobalQueue.post_thread.start()
