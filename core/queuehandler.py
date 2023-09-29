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

    #def get_prompt(self):
    #    return self.prompt


# the queue object for Deforum command
class DeforumObject:
    def __init__(self, cog, ctx, deforum_settings, view):
        self.cog = cog
        self.ctx = ctx
        self.deforum_settings = deforum_settings
        self.view = view
        self.is_done = False

    #def get_prompt(self):
    #    return self.deforum_settings["Prompts"]


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
    def __init__(self, cog, ctx, prompt, num_prompts, max_length, temperature, top_k, repetition_penalty):
        self.cog = cog
        self.ctx = ctx
        self.prompt = prompt
        self.num_prompts = num_prompts
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty


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
                item_info = f"\n{index}. {display_names.get(item.__class__.__name__, item.__class__.__name__)}"  # Utilisation du mapping
                if isinstance(item, DrawObject):
                    item_info += f" - Prompt: {item.prompt[:30] + '...' if len(item.prompt) > 30 else item.prompt}"
                elif isinstance(item, DeforumObject):
                    first_prompt = item.deforum_settings["Prompts"][0] if item.deforum_settings["Prompts"] else ""
                    item_info += f" - `{first_prompt[:30] + '...`' if len(first_prompt) > 30 else first_prompt}"
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
    async def update_progress_message(queue_object):
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

        # progress loop
        null_counter = 0
        while not queue_object.is_done:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:7860/sdapi/v1/progress?skip_current_image=false") as response:
                    data = await response.json()

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
                    #prompt = queue_object.get_prompt()s
                    #short_prompt = prompt[:125] + "..." if len(prompt) > 125 else prompt
                    short_prompt = queue_object.prompt[:125] + "..." if len(prompt) > 125 else prompt
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

                    await asyncio.sleep(1)

        # done, delete
        await progress_msg.delete()

    def process_queue():
        def start(target_queue: list[DrawObject | UpscaleObject | IdentifyObject | GenerateObject | DeforumObject]):
            queue_object = target_queue.pop(0)
            queue_object.cog.dream(GlobalQueue.event_loop, queue_object)

        if GlobalQueue.queue:
            start(GlobalQueue.queue)

    '''
    async def update_progress_message_generate(instance, queue_object, num_prompts):
        ctx = queue_object.ctx

        # check for an existing progression message, if yes delete the previous one
        async for old_msg in ctx.channel.history(limit=25):
            if old_msg.embeds:
                if old_msg.embeds[0].title == "──── Running Job Progression ────":
                    await old_msg.delete()

        # send first message to discord, Initialization
        embed = discord.Embed(title="Initialization...", color=discord.Color.blue())
        progress_msg = await ctx.send(embed=embed)

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

            # update the leaderboard
            LeaderboardCog.update_leaderboard(queue_object.ctx.author.id, str(queue_object.ctx.author), "Generate_Count")

            # update progress message
            description = f"Generating {num_prompts} {'prompt' if num_prompts == 1 else 'prompts'}!"
            description += f"\nCurrently on prompt {i+1} of {num_prompts}."
            embed = discord.Embed(title=f"──── Running Job Progression ────", description=description, color=discord.Color.random())
            await progress_msg.edit(embed=embed)

        # check if the message has been moved in the chat and move it down if needed
        while not queue_object.is_done:
            latest_message = await ctx.channel.history(limit=1).flatten()
            latest_message = latest_message[0] if latest_message else None

            if latest_message and latest_message.id != progress_msg.id:
                await progress_msg.delete()
                progress_msg = await ctx.send(embed=embed)
            
            await asyncio.sleep(0.5)

        # done, delete
        await progress_msg.delete()
        '''


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
