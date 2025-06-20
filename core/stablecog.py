import asyncio
from asyncio import run_coroutine_threadsafe
import base64
import csv
import discord
import io
import math
import os
import random
import requests
import time
import traceback
#from core.mask_server import MaskEditorServer
from PIL import Image, PngImagePlugin
from discord import option, OptionChoice
from discord.ext import commands
from typing import Optional
from core import queuehandler
from core import viewhandler
from core import settings
from core import settingscog
from core.queuehandler import GlobalQueue
from core.leaderboardcog import LeaderboardCog
from core.color_correction_sharpening import apply_color_correction
from core.persistence import save_message, load_all, delete_message

USE_LLAMA_CPP = True

if USE_LLAMA_CPP:
    from llama_cpp import Llama
else:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


# ratios dic
size_ratios = {
    "Portrait: 2:3 - 832x1216": (832, 1216),
    "Landscape: 3:2 - 1216x832": (1216, 832),
    "Fullscreen: 4:3 - 1152x896": (1152, 896),
    "Widescreen: 16:9 - 1344x768": (1344, 768),
    "Ultrawide: 21:9 - 1536x640": (1536, 640),
    "Square: 1:1 - 1024x1024": (1024, 1024),
    "Tall: 9:16 - 768x1344": (768, 1344)
}


class GPT2ModelSingleton:
    _instance = None
    lock = asyncio.Lock()
    llm = None
    model = None
    tokenizer = None
    pipe = None
    model_name = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._load_model()
        return cls._instance

    @classmethod
    def _load_model(cls):
        if USE_LLAMA_CPP:
            model_path = "core/WizzGPT6/WizzGPTv6.Q8_0.gguf"
            cls.llm = Llama(
                model_path=model_path,
                n_ctx=1024,
                n_threads=8,
                use_mlock=True,
                verbose=False,
            )
        else:
            model_path = "core/WizzGPT6"
            cls.tokenizer = AutoTokenizer.from_pretrained(model_path)
            cls.model = AutoModelForCausalLM.from_pretrained(model_path)
            cls.pipe = pipeline(
                'text-generation',
                model=cls.model,
                tokenizer=cls.tokenizer,
                num_return_sequences=1,
                eos_token_id=cls.tokenizer.eos_token_id,
                max_length=90,
                temperature=1.25,
                top_p=0.92,
                top_k=40,
                no_repeat_ngram_size=5,
                repetition_penalty=1.4,
                early_stopping=True
            )

            cls.llm = cls.pipe

        if os.path.isdir(model_path):
            cls.model_name = os.path.basename(model_path)
        else:
            cls.model_name = os.path.basename(os.path.dirname(model_path))

infinite_flags = set()
infinite_enqueue_lock = asyncio.Lock()

class StableCog(commands.Cog, name='Stable Diffusion', description='Create images from natural language.'):
    ctx_parse = discord.ApplicationContext

    def __init__(self, bot, called_from_button=False):
        self.bot = bot
        self.pipe = None

    if len(settings.global_var.size_range) == 0:
        size_auto = discord.utils.basic_autocomplete(settingscog.SettingsCog.size_autocomplete)
    else:
        size_auto = None

    async def generate_prompt_async(self, prompt: str):
        gpt2_singleton = GPT2ModelSingleton.get_instance()
        llm = gpt2_singleton.llm
        lock = GPT2ModelSingleton.lock

        async with lock:
            loop = asyncio.get_running_loop()
            if USE_LLAMA_CPP:
                res = await loop.run_in_executor(
                    None,
                    lambda: llm(
                        prompt,
                        max_tokens=75,
                        temperature=1.25,
                        top_p=0.90,
                        #min_p=0.15,
                        top_k=48,
                        repeat_penalty=1.4
                    )
                )
                generated = res["choices"][0]["text"]
            else:
                res = await loop.run_in_executor(None, lambda: llm(prompt))
                if isinstance(res, list) and res and 'generated_text' in res[0]:
                    generated = res[0]['generated_text']
                else:
                    generated = str(res)

            return prompt + generated

    def get_random_word(self, filename):
        chosen_line = None
        try:
            with open(filename, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for count, row in enumerate(reader, start=1):
                    # With probability 1/count, choose the current row
                    if row and random.randrange(count) == 0:
                        chosen_line = row
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

        return chosen_line[0] if chosen_line else None

    @commands.Cog.listener()
    async def on_ready(self):
        self.bot.add_view(viewhandler.DrawView(self))

    @commands.slash_command(name='draw', description='Create an image')
    @option(
        'prompt',
        str,
        description='A prompt to condition the model with.',
        required=False,
    )
    @option(
        'random_prompt',
        str,
        description='Generate a random image from a random prompt.',
        required=False,
        choices=['True', 'Infinite']
    )
    @option(
        'negative_prompt',
        str,
        description='Negative prompts to exclude from output.',
        required=False,
    )
    @option(
        'data_model',
        str,
        description='Select the data model for image generation.',
        required=False,
        autocomplete=discord.utils.basic_autocomplete(settingscog.SettingsCog.model_autocomplete),
    )
    @option(
        'steps',
        int,
        description='The amount of steps to sample the model.',
        min_value=1,
        required=False,
    )
    @option(
        'width',
        int,
        description='Width of the generated image.',
        required=False,
        autocomplete=size_auto,
        choices=settings.global_var.size_range
    )
    @option(
        'height',
        int,
        description='Height of the generated image.',
        required=False,
        autocomplete=size_auto,
        choices=settings.global_var.size_range
    )
    @option(
        'size_ratio',
        str,
        description='Select a size ratio for image generation. This overrides width & height!',
        required=False,
        choices=list(size_ratios.keys())
    )
    @option(
        'guidance_scale',
        str,
        description='Classifier-Free Guidance scale.',
        required=False,
    )
    @option(
        'distilled_cfg_scale',
        str,
        description='Distilled CFG Guidance scale.',
        required=False,
    )
    @option(
        'sampler',
        str,
        description='The sampling method to use for generation.',
        required=False,
        autocomplete=discord.utils.basic_autocomplete(settingscog.SettingsCog.sampler_autocomplete),
    )
    @option(
        'scheduler',
        str,
        description='The schedule type to use for generation.',
        required=False,
        autocomplete=discord.utils.basic_autocomplete(settingscog.SettingsCog.scheduler_autocomplete),
    )
    @option(
        'seed',
        int,
        description='The seed to use for reproducibility.',
        required=False,
    )
    @option(
        'styles',
        str,
        description='Apply a predefined style to the generation.',
        required=False,
        autocomplete=discord.utils.basic_autocomplete(settingscog.SettingsCog.style_autocomplete),
    )
    @option(
        'random_style',
        str,
        description='Choose a style at random.',
        required=False,
        choices=["True"]
    )
    @option(
        'extra_net',
        str,
        description='Apply an extra network to influence the output. To set multiplier, add :# (# = 0.0 - 1.0)',
        required=False,
        autocomplete=discord.utils.basic_autocomplete(settingscog.SettingsCog.extra_net_autocomplete),
    )
    @option(
        'adetailer',
        str,
        description='Choose which details to improve: Faces, Hands, or both.',
        required=False,
        choices=['Faces', 'Hands', 'Faces+Hands', 'Details++']
    )
    #@option(
    #    'poseref',
    #    str,
    #    description='The pose reference image URL.',
    #    required=False,
    #)
    #@option(
    #    'ipadapter',
    #    str,
    #    description='The reference image URL for IPAdapter.',
    #    required=False,
    #)
    @option(
        'highres_fix',
        str,
        description='Tries to fix issues from generating high-res images. Recommended: 4x-UltraMix_Balanced.',
        required=False,
        autocomplete=discord.utils.basic_autocomplete(settingscog.SettingsCog.hires_autocomplete),
    )
    @option(
        'clip_skip',
        int,
        description='Number of last layers of CLIP model to skip.',
        required=False,
        choices=[x for x in range(1, 13, 1)]
    )
    @option(
        'strength',
        str,
        description='The amount in which init_image will be altered (0.0 to 1.0).'
    )
    @option(
        'init_image',
        discord.Attachment,
        description='The starter image for generation. Remember to set strength value!',
        required=False,
    )
    @option(
        'init_url',
        str,
        description='The starter URL image for generation. This overrides init_image!',
        required=False,
    )
    @option(
        'batch',
        str,
        description='The number of images to generate. Batch format: count,size',
        required=False,
    )
    async def dream_handler(self, ctx: discord.ApplicationContext, *,
                            prompt: str,
                            random_prompt: Optional[bool] = False,
                            negative_prompt: str = None,
                            data_model: Optional[str] = None,
                            steps: Optional[int] = None,
                            width: Optional[int] = None, 
                            height: Optional[int] = None,
                            size_ratio: Optional[str] = None,
                            guidance_scale: Optional[str] = None,
                            distilled_cfg_scale: Optional[str] = None,
                            sampler: Optional[str] = None,
                            scheduler: Optional[str] = None,
                            seed: Optional[int] = -1,
                            styles: Optional[str] = None,
                            random_style: Optional[bool] = False,
                            extra_net: Optional[str] = None,
                            adetailer: Optional[bool] = None,
                            highres_fix: Optional[str] = None,
                            clip_skip: Optional[int] = None,
                            strength: Optional[str] = None,
                            init_image: Optional[discord.Attachment] = None,
                            init_url: Optional[str] = None,
                            #poseref: Optional[discord.Attachment] = None,
                            #ipadapter: Optional[discord.Attachment] = None,
                            batch: Optional[str] = None):

        called_from_button = getattr(ctx, 'called_from_button', False)

        # check if one of prompt or random_prompt option is enabled
        if not prompt and not random_prompt:
            await ctx.respond("Please provide a prompt or enable random prompt generation.", ephemeral=True)
            return

        if random_prompt == "Infinite":
            # on prévient l'utilisateur et on démarre la loop
            await ctx.respond("⏯️ Infinite random generation started. Use `/stopdraw` to stop.")
            infinite_flags.add(ctx.author.id)
            loop_opts = {
                "random_style": random_style,
                "negative_prompt": negative_prompt,
                "data_model":      data_model,
                "steps":           steps,
                "width":           width,
                "height":          height,
                "size_ratio":      size_ratio,
                "guidance_scale":  guidance_scale,
                "distilled_cfg_scale": distilled_cfg_scale,
                "sampler":         sampler,
                "scheduler":       scheduler,
                "styles":          styles,
                "extra_net":       extra_net,
                "adetailer":       adetailer,
                "highres_fix":     highres_fix,
                "clip_skip":       clip_skip,
                "strength":        strength,
                "init_image":      init_image,
                "init_url":        init_url,
                #"poseref":         poseref,
                #"ipadapter":       ipadapter,
                "batch":           batch
            }
            # lance en arrière-plan la boucle infinie AVEC ces options
            asyncio.create_task(self._infinite_loop(ctx, **loop_opts))
            return

        # generate a random prompt if random_prompt is True
        deferred = False
        if random_prompt == "True":
            await ctx.defer()            
            num_prompts = 1

            # manage True:x
            #if ':' in random_prompt:
            #    try:
            #        _, num_str = random_prompt.split(':', 1)
            #        num_prompts = int(num_str)

            #        if num_prompts < 1 or num_prompts > 10:
            #            await ctx.respond("Le nombre de prompts doit être entre 1 et 10.", ephemeral=True)
            #            return
            #    except ValueError:
            #        await ctx.respond("Invalid format for random prompt number. Use /draw random_prompt:true:x where x is a number.", ephemeral=True)
            #        return
            
            for _ in range(num_prompts):
                start_prompt = self.get_random_word('resources/random_prompts.csv')
                generated_prompt_task = asyncio.create_task(self.generate_prompt_async(start_prompt))
                generated_prompt = await generated_prompt_task
                prompt = generated_prompt
                deferred = True

        if random_style == "True":
            settings_cog = self.bot.get_cog('SettingsCog')
            if settings_cog:
                style_dict = settings_cog.get_available_styles()
                chosen_style_key = random.choice(list(style_dict.keys())) if style_dict else None
                if chosen_style_key:
                    styles = chosen_style_key

        # update defaults with any new defaults from settingscog
        channel = '% s' % ctx.channel.id
        settings.check(channel)
        if negative_prompt is None:
            negative_prompt = settings.read(channel)['negative_prompt']
        if steps is None:
            steps = settings.read(channel)['steps']
        if width is None:
            width = settings.read(channel)['width']
        if height is None:
            height = settings.read(channel)['height']
        if guidance_scale is None:
            guidance_scale = settings.read(channel)['guidance_scale']
        if distilled_cfg_scale is None:
            distilled_cfg_scale = settings.read(channel)['distilled_cfg_scale']
        if sampler is None:
            sampler = settings.read(channel)['sampler']
        if scheduler is None:
            scheduler = settings.read(channel)['scheduler']
        if styles is None:
            styles = settings.read(channel)['style']
        if highres_fix is None:
            highres_fix = settings.read(channel)['highres_fix']
        if clip_skip is None:
            clip_skip = settings.read(channel)['clip_skip']
        if strength is None:
            strength = settings.read(channel)['strength']
        if batch is None:
            batch = settings.read(channel)['batch']

        # if a model is not selected, do nothing
        model_name = 'Default'
        if data_model is None:
            data_model = settings.read(channel)['data_model']

        simple_prompt = prompt
        # run through mod function if any moderation values are set in config
        clean_negative = negative_prompt
        if settings.global_var.prompt_ban_list or settings.global_var.prompt_ignore_list or settings.global_var.negative_prompt_prefix:
            mod_results = settings.prompt_mod(simple_prompt, negative_prompt)
            if mod_results[0] == "Stop":
                await ctx.respond(f"I'm not allowed to draw the word {mod_results[1]}!", ephemeral=True)
                return
            if mod_results[0] == "Mod":
                if settings.global_var.display_ignored_words == "False":
                    simple_prompt = mod_results[1]
                prompt = mod_results[1]
                negative_prompt = mod_results[2]
                clean_negative = mod_results[3]

        # take selected data_model and get model_name, then update data_model with the full name
        for model in settings.global_var.model_info.items():
            if model[0] == data_model:
                model_name = model[0]
                data_model = model[1][0]
                # look at the model for activator token and prepend prompt with it
                if model[1][3]:
                    prompt = model[1][3] + " " + prompt
                break

        net_multi = 0.85
        if extra_net is not None:
            prompt, extra_net, net_multi = settings.extra_net_check(prompt, extra_net, net_multi)
        prompt = settings.extra_net_defaults(prompt, channel)

        if data_model != '':
            print(f'/Draw request -- {ctx.author.name} -- Prompt: {prompt}')
        else:
            print(f'/Draw request -- {ctx.author.name} -- Prompt: {prompt} -- Using model: {data_model}')

        if seed == -1:
            seed = random.randint(0, 0xFFFFFFFF)

        # url *will* override init image for compatibility, can be changed here
        if init_url:
            try:
                init_image = requests.get(init_url)
            except(Exception,):
                await ctx.send_response('URL image not found!\nI will do my best without it!')

        # size_ratio preset will override height and width
        if size_ratio:
            width, height = size_ratios.get(size_ratio, (width, height))

        # verify values and format aiya initial reply
        reply_adds = ''
        if (width != 512) or (height != 512):
            reply_adds += f' - Size: ``{width}``x``{height}``'
        reply_adds += f' - Seed: ``{seed}``'

        # lower step value to the highest setting if user goes over max steps
        if steps > settings.read(channel)['max_steps']:
            steps = settings.read(channel)['max_steps']
            reply_adds += f'\nExceeded maximum of ``{steps}`` steps! This is the best I can do...'
        if model_name != 'Default':
            if random_prompt in ("True", "Infinite"):
                gpt2_model_singleton = GPT2ModelSingleton.get_instance()
                reply_adds += f'\nModel: ``{model_name}`` (GPT2: {gpt2_model_singleton.model_name})'
            else:
                reply_adds += f'\nModel: ``{model_name}``'
        if clean_negative != settings.read(channel)['negative_prompt']:
            reply_adds += f'\nNegative Prompt: ``{clean_negative}``'
        if guidance_scale != settings.read(channel)['guidance_scale']:
            try:
                guidance_scale = float(str(guidance_scale).replace(",", "."))
                reply_adds += f'\nGuidance Scale: ``{guidance_scale}``'
            except Exception:
                reply_adds += f"\nGuidance Scale can't be ``{guidance_scale}``! Setting to default of `5.5`."
                guidance_scale = 5.5
        if distilled_cfg_scale != settings.read(channel)['distilled_cfg_scale']:
            try:
                distilled_cfg_scale = float(str(distilled_cfg_scale).replace(",", "."))
                reply_adds += f'\nDistilled CFG Scale: ``{distilled_cfg_scale}``'
            except Exception:
                reply_adds += f"\nDistilled CFG Scale can't be ``{distilled_cfg_scale}``! Setting to default of `3.5`."
                distilled_cfg_scale = 3.5
        if sampler != settings.read(channel)['sampler']:
            reply_adds += f'\nSampler: ``{sampler}``'
        if scheduler != settings.read(channel)['scheduler']:
            reply_adds += f' - Scheduler: ``{scheduler}``'
        if init_image:
            # try to convert string to Web UI-friendly float
            try:
                strength = strength.replace(",", ".")
                float(strength)
                reply_adds += f'\nStrength: ``{strength}``'
            except(Exception,):
                reply_adds += f"\nStrength can't be ``{strength}``! Setting to default of `0.5`."
                strength = 0.5
            reply_adds += f'\nURL Init Image: ``{init_image.url}``'
        if adetailer is not None:
            reply_adds += f'\nADetailer: ``{adetailer}``'
        if highres_fix != 'Disabled':
            reply_adds += f'\nHighres Fix: ``{highres_fix}``'
        # try to convert batch to usable format
        batch_check = settings.batch_format(batch)
        batch = list(batch_check)
        if batch[0] != 1 or batch[1] != 1:
            max_batch = settings.batch_format(settings.read(channel)['max_batch'])
            # if only one number is provided, try to generate the requested amount, prioritizing batch size
            if batch[2] == 1:
                # if over the limits, cut the number in half and let AIYA scale down
                total = max_batch[0] * max_batch[1]
                if batch[0] > total:
                    batch[0] = math.ceil(batch[0] / 2)
                    batch[1] = math.ceil(batch[0] / 2)
                else:
                    # do... math
                    difference = math.ceil(batch[0] / max_batch[1])
                    multiple = int(batch[0] / difference)
                    new_total = difference * multiple
                    requested = batch[0]
                    batch[0], batch[1] = difference, multiple
                    if requested % difference != 0:
                        reply_adds += f"\nI can't draw exactly ``{requested}`` pictures! Settling for ``{new_total}``."
            # check batch values against the maximum limits
            if batch[0] > max_batch[0]:
                reply_adds += f"\nThe max batch count I'm allowed here is ``{max_batch[0]}``!"
                batch[0] = max_batch[0]
            if batch[1] > max_batch[1]:
                reply_adds += f"\nThe max batch size I'm allowed here is ``{max_batch[1]}``!"
                batch[1] = max_batch[1]
            reply_adds += f'\nBatch count: ``{batch[0]}`` - Batch size: ``{batch[1]}``'
        if styles != settings.read(channel)['style']:
            reply_adds += f'\nStyle: ``{styles}``'
        if extra_net is not None and extra_net != 'None':
            reply_adds += f'\nExtra network: ``{extra_net}``'
            if net_multi != 0.85:
                reply_adds += f' (multiplier: ``{net_multi}``)'
        if clip_skip != settings.read(channel)['clip_skip']:
            reply_adds += f'\nCLIP skip: ``{clip_skip}``'
        #if poseref is not None:
        #    reply_adds += f'\nPose Reference URL: ``{poseref}``'
        #if ipadapter is not None:
        #    reply_adds += f'\nIPAdapter Reference URL: ``{ipadapter}``'

        epoch_time = int(time.time())

        # set up tuple of parameters to pass into the Discord view
        input_tuple = (
            ctx, simple_prompt, prompt, negative_prompt, data_model, steps, width, height, guidance_scale, sampler, seed, strength,
            init_image, batch, styles, highres_fix, clip_skip, extra_net, epoch_time, adetailer, scheduler, distilled_cfg_scale)# poseref, ipadapter

        view = viewhandler.DrawView(input_tuple)
        # setup the queue
        user_queue_limit = settings.queue_check(ctx.author)
        if queuehandler.GlobalQueue.dream_thread.is_alive():
            if user_queue_limit == "Stop":
                await ctx.send_response(content=f"Please wait! You're past your queue limit of {settings.global_var.queue_limit}.", ephemeral=True)
            else:
                queuehandler.GlobalQueue.queue.append(queuehandler.DrawObject(self, *input_tuple, view))
        else:
            await queuehandler.process_dream(self, queuehandler.DrawObject(self, *input_tuple, view))

        message_to_send = f'<@{ctx.author.id}>, {settings.messages()}\nQueue: ``{len(queuehandler.GlobalQueue.queue)}`` - ``{simple_prompt}``\nSteps: ``{steps}``{reply_adds}'

        # check if webui is online
        webui_is_offline = settings.check_webui_running(settings.global_var)
        if webui_is_offline:
            message_to_send += "\nNote: The model is currently offline. Your request won't be lost, it will be processed when it's back online !"

        # check prompts length
        base_message_length = len(f'<@{ctx.author.id}>, {settings.messages()}\nQueue: ``{len(queuehandler.GlobalQueue.queue)}`` - ````\nSteps: ``{steps}``{reply_adds}')
        available_length_for_prompts = 2000 - base_message_length

        # truncate prompt if needed
        if len(prompt) + len(negative_prompt) > available_length_for_prompts:
            total_prompt_length = len(prompt) + len(negative_prompt)
            prompt_ratio = len(prompt) / total_prompt_length
            negative_prompt_ratio = len(negative_prompt) / total_prompt_length

            prompt_length_limit = int(available_length_for_prompts * prompt_ratio)
            negative_prompt_length_limit = int(available_length_for_prompts * negative_prompt_ratio)

            prompt = prompt[:prompt_length_limit - 5] + "..." if len(prompt) > prompt_length_limit else prompt
            negative_prompt = negative_prompt[:negative_prompt_length_limit - 5] + "..." if len(negative_prompt) > negative_prompt_length_limit else negative_prompt

        # reconstruc with truncated prompts
        message_to_send = f'<@{ctx.author.id}>, {settings.messages()}\nQueue: ``{len(queuehandler.GlobalQueue.queue)}`` - ``{prompt}``\nSteps: ``{steps}``{reply_adds}'
        #if negative_prompt:
        #    message_to_send += f'\nNegative Prompt: ``{negative_prompt}``'
         
        # send to discord
        if called_from_button:
            await ctx.channel.send(message_to_send)
        elif getattr(ctx, "_infinite_job", False):
            await ctx.channel.send(message_to_send)
        else:
            if not ctx.interaction.response.is_done():
                await ctx.respond(message_to_send)
            else:
                await ctx.followup.send(message_to_send)

    @commands.slash_command(name='stopdraw', description='Stop infinite random generation')
    @commands.guild_only()
    async def stop_draw(self, ctx: discord.ApplicationContext):
        user_id = ctx.author.id
        if user_id in infinite_flags:
            infinite_flags.remove(user_id)
            await ctx.respond("🛑 Infinite generation stopped !")
        else:
            await ctx.respond("❌ No Infinite generation to stop.", ephemeral=True)

    # === MODIF : helper pour la boucle infinie
    async def _infinite_loop(self, ctx: discord.ApplicationContext, **opts):
        try:
            while ctx.author.id in infinite_flags:
                start_prompt = self.get_random_word('resources/random_prompts.csv')
                generated_prompt = await self.generate_prompt_async(start_prompt)
                async with infinite_enqueue_lock:
                    ctx._infinite_job = True
                    await self.dream_handler(
                        ctx,
                        prompt=generated_prompt,
                        random_prompt=None,
                        **opts
                    )
                    delattr(ctx, "_infinite_job")

                while queuehandler.GlobalQueue.dream_thread.is_alive():
                    await asyncio.sleep(3)
                print("Infinite job done, next one will start soon\n")

                await asyncio.sleep(1)

        except Exception as e:
            print(f"[InfiniteLoop] erreur pour {ctx.author.id} : {e}")
        finally:
            # s’assure qu’on enlève le flag si erreur ou fin
            infinite_flags.discard(ctx.author.id)


    # the function to queue Discord posts
    def post(self, event_loop: queuehandler.GlobalQueue.post_event_loop, post_queue_object: queuehandler.PostObject):
        async def send_message():
            embed = post_queue_object.embed
            # Patch: always pass a valid embed (discord.Embed or list or None)
            if embed is not None and not isinstance(embed, (discord.Embed, list)):
                embed = None
            try:
                await post_queue_object.ctx.channel.send(
                    content=post_queue_object.content,
                    file=post_queue_object.file,
                    embed=embed,
                    view=post_queue_object.view
                )
            except discord.HTTPException as e:
                if e.code == 40005 or "Payload Too Large" in str(e):
                    file = post_queue_object.file
                    file_size_mb = None
                    if file and hasattr(file, "fp"):
                        try:
                            fp = file.fp
                            if hasattr(fp, "getbuffer"):
                                file_size_mb = len(fp.getbuffer()) / (1024 * 1024)
                            elif hasattr(fp, "name") and os.path.exists(fp.name):
                                file_size_mb = os.path.getsize(fp.name) / (1024 * 1024)
                        except Exception:
                            file_size_mb = None
                    mb = f"{file_size_mb:.2f}" if file_size_mb is not None else "unknown"
                    await post_queue_object.ctx.channel.send(
                        f"❌ Failed to send image: file size is {mb} MB, but it exceeds the server's allowed file size limit."
                    )
                else:
                    await post_queue_object.ctx.channel.send(
                        f"❌ An error occurred while sending the image: {str(e)}"
                    )

        event_loop.create_task(send_message())

        if queuehandler.GlobalQueue.post_queue:
            self.post(self.event_loop, self.queue.pop(0))

    # generate the image
    def dream(self, event_loop: queuehandler.GlobalQueue.event_loop, queue_object: queuehandler.DrawObject):
    
        # start progression message
        run_coroutine_threadsafe(GlobalQueue.update_progress_message(queue_object), event_loop)

        try:
            start_time = time.time()

            # construct a payload for data model, then the normal payload
            model_payload = {
                "sd_model_checkpoint": queue_object.data_model
            }

            #style_to_use = queue_object.styles
            #if "zavyyumexl" in queue_object.data_model.lower():
            #    style_to_use = "Yume Style"
            #    queue_object.negative_prompt = ""
            #    queue_object.sampler = "Euler a"

            payload = {
                "prompt": queue_object.prompt,
                "negative_prompt": queue_object.negative_prompt,
                "steps": queue_object.steps,
                "width": queue_object.width,
                "height": queue_object.height,
                "cfg_scale": queue_object.guidance_scale,
                "distilled_cfg_scale": queue_object.distilled_cfg_scale,
                "sampler_name": queue_object.sampler,
                "scheduler": queue_object.scheduler,
                "seed": queue_object.seed,
                "seed_resize_from_h": -1,
                "seed_resize_from_w": -1,
                "denoising_strength": None,
                "n_iter": queue_object.batch[0],
                "batch_size": queue_object.batch[1],
                "styles": [
                    queue_object.styles
                ]
            }

            # update payload if init_img or init_url is used
            if queue_object.init_image is not None:
                image = base64.b64encode(requests.get(queue_object.init_image.url, stream=True).content).decode('utf-8')
                img_payload = {
                    "init_images": [
                        'data:image/png;base64,' + image
                    ],
                    "denoising_strength": queue_object.strength
                }
                payload.update(img_payload)
                
            # if Details++ then use hires
            if queue_object.adetailer == 'Details++' and queue_object.highres_fix == 'Disabled':
                channel_id = str(queue_object.ctx.channel.id)
                queue_object.highres_fix = settings.read(channel_id)['upscaler_1']

            # hires payload
            if queue_object.highres_fix != 'Disabled':
                upscale_ratio = 1.6
                queue_object.width = int(queue_object.width * upscale_ratio)
                queue_object.height = int(queue_object.height * upscale_ratio)
                highres_payload = {
                    "enable_hr": True,
                    "hr_upscaler": queue_object.highres_fix,
                    "hr_scale": upscale_ratio,
                    "hr_cfg": float(queue_object.guidance_scale),
                    "hr_distilled_cfg": float(queue_object.distilled_cfg_scale),
                    "hr_second_pass_steps": int(queue_object.steps / 1.3),
                    "denoising_strength": 0.48, #queue_object.strength, # 0.48s
                    "hr_prompt": "(Sharp focus:2), " + queue_object.prompt,
                    "hr_negative_prompt": "(Undersaturated, washed colors), (blurry), (poorly drawn:2), " + queue_object.negative_prompt,
                    #"hr_prompt": "(subsurface scattering:2), (extremely fine details:2), (consistency:2), smooth, round pupils, perfect teeth, perfect hands, (extremely detailed teeth:2), (extremely detailed hands:2), (extremely detailed face:2), (extremely detailed eyes:2), photorealism, film grain, candid camera, color graded cinematic, eye catchlights, atmospheric lighting, shallow dof, " + queue_object.prompt,
                    #"hr_negative_prompt": "(low quality:2), (worst quality:2), (bad hands:2), (ugly eyes:2), (fused fingers:2), (elongated fingers:2), (additionnal fingers:2), missing fingers, long nails, grainy, (intricated patterns:2), (intricated vegetation:2), grainy, lowres, noise, poor detailing, unprofessional, unsmooth, license plate, aberrations, collapsed, conjoined, extra windows, harsh lighting, multiple levels, overexposed, rotten, sketchy, twisted, underexposed, unnatural, unreal engine, unrealistic, video game, (poorly rendered face:2), " + queue_object.negative_prompt
                    "hr_additional_modules": ["Use same choices"],
                }
                payload.update(highres_payload)

            # add any options that would go into the override_settings
            override_settings = {"CLIP_stop_at_last_layers": queue_object.clip_skip}

            alwayson_scripts_settings = {}
            
            # add adetailer settings
            if queue_object.adetailer and queue_object.adetailer != "None":
                model_mappings = {
                    "Faces": {
                        "ad_model": "face_yolov8s.pt",
                        "ad_use_inpaint_width_height": True,
                        "ad_inpaint_width": 1024,
                        "ad_inpaint_height": 1024,
                        "ad_denoising_strength": 0.40,
                        "ad_dilate_erode": 4,
                        "ad_mask_max_ratio": 0.25,
                        "ad_mask_blur": 4,
                        "ad_inpaint_only_masked": True,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_use_noise_multiplier": True,
                        "ad_noise_multiplier": 0.85,
                        "ad_prompt": "(extremely detailed face), (fine detailed eyes), raytracing, subsurface scattering, hyperrealistic, extreme skin details, skin pores, deep shadows, subsurface scattering, amazing textures, filmic, macro, shallow dof, shallow depth of field, beautiful eyes, extremely detailed pupil, " + queue_object.prompt,
                        "ad_negative_prompt": "(low quality:2), (asymmetric eyes, bad eyes:2), lowres, (heterochromia:2)"
                    },
                    "Hands": {
                        "ad_model": "hand_yolov8n.pt",
                        "ad_use_inpaint_width_height": True,
                        "ad_inpaint_width": 1024,
                        "ad_inpaint_height": 1024,
                        "ad_denoising_strength": 0.45,
                        "ad_dilate_erode": 4,
                        "ad_mask_max_ratio": 0.15,
                        "ad_mask_blur": 4,
                        "ad_inpaint_only_masked": True,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_use_noise_multiplier": True,
                        "ad_noise_multiplier": 0.85,
                        "ad_prompt": "(extremely detailed hand), (extremely detailed fingers), natural nails color, " + queue_object.prompt,
                        "ad_negative_prompt": "(low quality:2), (malformed:2), lowres, colored nails, undetailed hand, fused fingers, elongated fingers, wrong hand anatomy, additionnal fingers, missing fingers, inversed hand"
                        #"ad_controlnet_module": "openpose_full",
                        #"ad_controlnet_model": "control_openpose-fp16 [72a4faf9]"
                    }
                }
                args = [True]

                if queue_object.adetailer == "Faces+Hands" or queue_object.adetailer == "Details++":
                    args.extend([model_mappings["Faces"], model_mappings["Hands"]])
                #elif queue_object.adetailer == "Details++":
                #    args.extend([model_mappings["Faces"], model_mappings["Hands"]])#, model_mappings["Details"]])
                else:
                    args.append(model_mappings[queue_object.adetailer])

                alwayson_scripts_settings = {
                    "ADetailer": {
                        "args": args
                    }
                }

            controlnet_args = []

            # Vérification et ajout de la configuration pour poseref
            #if queue_object.poseref is not None:
            #    pimage = base64.b64encode(requests.get(queue_object.poseref, stream=True).content).decode('utf-8')
            #    poseref_payload = {
            #        "input_image": 'data:image/png;base64,' + pimage,
            #        "control_mode": "Balanced",
            #        "pixel_perfect": True,
            #        "loopback": False,
            #        "low_vram": True,
            #        "module": "openpose_full",
            #        "model": "control_openpose-fp16 [72a4faf9]",
            #        "resize_mode": "Resize and Fill",
            #        "weight": 1,
            #        "preprocessor_res": 768
            #    }
            #    controlnet_args.append(poseref_payload)

            # Vérification et ajout de la configuration pour ipadapter
           # if queue_object.ipadapter is not None:
           #     pimage = base64.b64encode(requests.get(queue_object.ipadapter, stream=True).content).decode('utf-8')
           #     ipadapter_payload = {
           #         "input_image": 'data:image/png;base64,' + pimage,
           #         "control_mode": "My prompt is more important",
           #         "pixel_perfect": True,
           #         "loopback": False,
           #         "low_vram": True,
           #         "module": "ip-adapter_clip_sdxl",
           #         "model": "IpAdapter [af81326a]",
           #         "resize_mode": "Resize and Fill",
           #         "weight": 0.70 #,
           #         #"preprocessor_res": 768,
           #         #"guidance_start": 0.0,
           #         #"guidance_end": 1.0
           #s     }
           #     controlnet_args.append(ipadapter_payload)

            # Ajout des configurations controlnet au alwayson_scripts_settings s'il y a des éléments dans controlnet_args
            if controlnet_args:
                alwayson_scripts_settings["controlnet"] = {
                    "args": controlnet_args
                }

            # update payload with override_settings
            override_payload = {
                "override_settings": override_settings
            }
            payload.update(override_payload)

            # update payload with alwayson_scripts
            alwayson_scripts_payload = {
                "alwayson_scripts": alwayson_scripts_settings
            }
            payload.update(alwayson_scripts_payload)

            # send normal payload to webui and only send model payload if one is defined
            s = settings.authenticate_user()

            if queue_object.data_model != '':
                try:
                    s.post(url=f'{settings.global_var.url}/sdapi/v1/options', json=model_payload)
                except requests.exceptions.ConnectionError:
                    print("Connection error. No response from API. (StableCog l.756)")

            is_flux = "flux" in queue_object.data_model.lower()

            # Gérer le preset
            forge_preset = "flux" if is_flux else "sdxl"

            # Gérer le storage dtype
            if "nf4" in queue_object.data_model.lower():
                forge_unet_storage_dtype = "bnb-fp4 (fp16 LoRA)"
            else:
                forge_unet_storage_dtype = "Automatic (fp16 LoRA)"

            # Modules à charger
            modules_dir = "C:\\Users\\wizz\\stable-diffusion\\stable-diffusion-webui-forge\\models\\text_encoder\\"
            modules_to_load = [
                modules_dir + "ViT-L-14-REG-GATED-balanced-ckpt12.safetensors"
            ]
            if is_flux:
                modules_to_load += [
                    modules_dir + "t5xxl_fp16.safetensors",
                    modules_dir + "flux_vae.safetensors"
                ]
            else:
                modules_to_load += [
                    modules_dir + "sdxl_vae.safetensors"
                ]

            # Construire le payload options complet
            forge_options_payload = {
                "forge_preset": forge_preset,
                "forge_additional_modules": modules_to_load,
                "forge_unet_storage_dtype": forge_unet_storage_dtype,
                "img2img_extra_noise": 0.015 if is_flux else 0.045
            }

            # Envoi la configuration à /options
            try:
                s.post(url=f'{settings.global_var.url}/sdapi/v1/options', json=forge_options_payload)
            except requests.exceptions.ConnectionError:
                print("Connection error. No response from API pour forge options.")

            if queue_object.init_image is not None:
                response = s.post(url=f'{settings.global_var.url}/sdapi/v1/img2img', json=payload)
            else:
                response = s.post(url=f'{settings.global_var.url}/sdapi/v1/txt2img', json=payload)

            response_data = response.json()

            # Ultimate SD Upscale payload
            if queue_object.adetailer == 'Details++' and response.ok:
                generated_images = response_data.get("images")
                upscaled_images_data = []
                upscaled_images_metadata = []

                # adjust values
                custom_scale, denoising_strength = (2.4, 0.42) if queue_object.adetailer == 'Details++' else (1, 0.10)
                tile_width = int(queue_object.width * custom_scale) / 3 if queue_object.highres_fix != 'Disabled' else int(queue_object.width * custom_scale)
                tile_height = int(queue_object.height * custom_scale) / 3 if queue_object.highres_fix != 'Disabled' else int(queue_object.height * custom_scale)
                queue_object.width = int(queue_object.width * custom_scale)
                queue_object.height = int(queue_object.height * custom_scale)

                # update the Extra Noise setting
                response = requests.get(url=f'{settings.global_var.url}/sdapi/v1/options')

                is_flux = "flux" in queue_object.data_model.lower()

                # Adapt noise values to model type
                if is_flux:
                    default_extra_noise = 0.015
                    default_initial_noise = 1
                else:
                    default_extra_noise = 0.045
                    default_initial_noise = 1

                if response.ok:
                    current_options = response.json()
                    original_extra_noise = current_options.get("img2img_extra_noise", default_extra_noise)
                    original_initial_noise = current_options.get("initial_noise_multiplier", default_initial_noise)
                    print(f"Original Extra Noise to restore after Details++: {original_extra_noise}")
                    print(f"Original Initial Noise to restore after Details++: {original_initial_noise}")
                else:
                    print("Error retrieving options")

                if is_flux:
                    option_payload = {"img2img_extra_noise": 0, "initial_noise_multiplier": 1.02}
                else:
                    option_payload = {"img2img_extra_noise": 0, "initial_noise_multiplier": 1.11}

                response = s.post(url=f'{settings.global_var.url}/sdapi/v1/options', json=option_payload)
                if response.ok:
                    print("Options updated successfully for Details++")
                else:
                    print("Error updating options")


                for index, generated_image_base64 in enumerate(generated_images):
                    original_image = Image.open(io.BytesIO(base64.b64decode(generated_image_base64)))
                    original_metadata = PngImagePlugin.PngInfo()
                    for k, v in original_image.info.items():
                        original_metadata.add_text(k, v)
                    upscaled_images_metadata.append(original_metadata)

                    # adjust steps
                    steps_as_int = int(queue_object.steps)
                    adjusted_steps = int(steps_as_int * 1.5)

                    upscale_payload = {
                        "prompt": "(Sharp focus:2), " + queue_object.prompt,
                        "negative_prompt": "(Undersaturated, washed colors), (blurry), (poorly drawn:2), " + queue_object.negative_prompt,
                        "steps": adjusted_steps,
                        "cfg_scale": queue_object.guidance_scale,
                        "sampler_name": queue_object.sampler,
                        "scheduler": queue_object.scheduler,
                        "seed": queue_object.seed,
                        "denoising_strength": denoising_strength,
                        "script_name": "ultimate sd upscale",
                        "script_args": [
                            None,  # _ (not used)
                            tile_width,  # tile_width
                            tile_height,  # tile_height
                            0,  # mask_blur
                            448,  # padding
                            64,  # seams_fix_width
                            0.30,  # seams_fix_denoise
                            256,  # seams_fix_padding
                            6,  # upscaler_index
                            True,  # save_upscaled_image a.k.a Upscaled
                            0,  # redraw_mode
                            False,  # save_seams_fix_image a.k.a Seams fix
                            0,  # seams_fix_mask_blur
                            0,  # seams_fix_type
                            1,  # target_size_type
                            queue_object.width,  # custom_width
                            queue_object.height,  # custom_height
                            custom_scale  # custom_scale
                        ],
                        "init_images": [
                            generated_image_base64
                        ]
                    }

                    soft_inpainting_payload = {
                        "Soft inpainting": True,
                        "Schedule bias": 0.45,           # Encore plus tôt → plus progressif
                        "Preservation strength": 0.14,   # Moins de préservation → plus de fondu/blend
                        "Transition contrast boost": 0.8,# Réduit drastiquement le contraste dans la zone de transition
                        "Mask influence": 0.5,           # Légèrement + d’importance au masque
                        "Difference threshold": 0.19,    # Plus sensible aux petites différences
                        "Difference contrast": 0.42,     # Encore + doux
                    }
                    upscale_payload["alwayson_scripts"] = {"soft inpainting": {"args": [soft_inpainting_payload]}}

                    # Details ++
                    if queue_object.adetailer == 'Details++':
                        combined_alwayson_scripts_payload = {
                            "ADetailer": {
                                "args": [
                                    True,
                                    False,
                                    {
                                        "ad_model": "face_yolov8s.pt",
                                        "ad_use_inpaint_width_height": True,
                                        "ad_inpaint_width": 1024,
                                        "ad_inpaint_height": 1024,
                                        "ad_denoising_strength": 0.36,
                                        "ad_noise_multiplier": 0.85,
                                        "ad_dilate_erode": 4,
                                        "ad_mask_max_ratio": 0.25,
                                        "ad_mask_blur": 4,
                                        "ad_inpaint_only_masked": True,
                                        "ad_inpaint_only_masked_padding": 64,
                                        "ad_x_offset": 24,
                                        "ad_y_offset": 24,
                                        "ad_prompt": "(extremely detailed face), (round pupils,  detailed eyes), raytracing, subsurface scattering, hyperrealistic, extreme skin details, skin pores, deep shadows, subsurface scattering, amazing textures, filmic, macro, shallow dof, shallow depth of field, beautiful eyes, extremely detailed pupil, " + queue_object.prompt,
                                        "ad_negative_prompt": "(low quality:2), (asymmetric eyes, bad eyes:2), lowres, (heterochromia:2)"
                                    },
                                    {
                                        "ad_model": "hand_yolov8n.pt",
                                        "ad_use_inpaint_width_height": True,
                                        "ad_inpaint_width": 1024,
                                        "ad_inpaint_height": 1024,
                                        "ad_denoising_strength": 0.45,
                                        "ad_noise_multiplier": 0.85,
                                        "ad_dilate_erode": 4,
                                        "ad_mask_max_ratio": 0.15,
                                        "ad_mask_blur": 4,
                                        "ad_inpaint_only_masked": True,
                                        "ad_inpaint_only_masked_padding": 64,
                                        "ad_x_offset": 24,
                                        "ad_y_offset": 24,
                                        #"ad_use_noise_multiplier": True,
                                        #"ad_noise_multiplier": 1.03,
                                        "ad_prompt": "(extremely detailed hand), (extremely detailed fingers), natural nails color, " + queue_object.prompt,
                                        "ad_negative_prompt": "(low quality:2), (malformed:2), lowres, colored nails, undetailed hand, fused fingers, elongated fingers, wrong hand anatomy, additionnal fingers, missing fingers, inversed hand"
                                    }#,
                                    #{
                                    #    "ad_model": "yolov8x-oiv7.pt",
                                    #    "ad_model_classes": "",
                                    #    "ad_use_inpaint_width_height": True,
                                    #    "ad_inpaint_width": 1024,
                                    #    "ad_inpaint_height": 1024,
                                    #    "ad_denoising_strength": 0.32,
                                    #    "ad_dilate_erode": 4,
                                    #    "ad_mask_max_ratio": 0.75,
                                    #    "ad_mask_blur": 4,
                                    #    "ad_inpaint_only_masked": True,
                                    #    "ad_inpaint_only_masked_padding": 96,
                                    #    #"ad_use_noise_multiplier": True,
                                    #    #"ad_noise_multiplier": 1.03,
                                    #    "ad_prompt": "(extremely detailed:2), " + queue_object.prompt,
                                    #    #"ad_negative_prompt": "(low quality:2), (malformed:2), lowres, colored nails, undetailed hand, fused fingers, elongated fingers, wrong hand anatomy, additionnal fingers, missing fingers, inversed hand"
                                    #}
                                ]
                            }
                        }

                        #if queue_object.highres_fix != 'Disabled':
                        #    soft_inpainting_payload = {
                        #        "Soft inpainting": True,
                        #        "Schedule bias": 1,
                        #        "Preservation strength": 0.5,
                        #        "Transition contrast boost": 4,
                        #        "Mask influence": 0,
                        #        "Difference threshold": 0.5,
                        #        "Difference contrast": 2,
                        #    }
                        #    combined_alwayson_scripts_payload["soft inpainting"] = {"args": [soft_inpainting_payload]}

                        #upscale_payload["alwayson_scripts"] = combined_alwayson_scripts_payload

                    # Send payload to img2img
                    upscale_response = s.post(url=f'{settings.global_var.url}/sdapi/v1/img2img', json=upscale_payload)
                    if upscale_response.ok:
                        upscale_response_data = upscale_response.json()
                        upscaled_images = upscale_response_data.get("images")
                        for upscaled_image_base64 in upscaled_images:
                            upscaled_image = Image.open(io.BytesIO(base64.b64decode(upscaled_image_base64)))
                            metadata = upscaled_images_metadata[index]
                            buffered = io.BytesIO()
                            upscaled_image.save(buffered, format="PNG", pnginfo=metadata)
                            upscaled_image_with_metadata_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                            upscaled_images_data.append(upscaled_image_with_metadata_base64)
                    else:
                        print("Error while upscaling with ultimate_sd_upscale")

                response_data["images"] = upscaled_images_data

                # restore the original extra noise
                restore_payload = {
                    "img2img_extra_noise": original_extra_noise,
                    "initial_noise_multiplier": original_initial_noise
                }
                response = requests.post(url=f'{settings.global_var.url}/sdapi/v1/options', json=restore_payload)
                if response.ok:
                    print(f"Options restored successfully to {original_extra_noise} & {original_initial_noise}")
                else:
                    print("Error restoring options")

            end_time = time.time()

            # create safe/sanitized filename
            keep_chars = (' ', '.', '_')
            file_name = "".join(c for c in queue_object.simple_prompt if c.isalnum() or c in keep_chars).rstrip()
            epoch_time = queue_object.epoch_time

            # save local copy of image and prepare PIL images
            image_data = response_data['images']
            if not image_data or len(image_data) == 0:
                print("[dream] No images generated in response_data['images']")
                # Optionally: send a Discord error message
                event_loop.create_task(queue_object.ctx.channel.send(
                    "❌ Image generation failed (no image was returned by the model)."
                ))
                queue_object.is_done = True
                return
            
            count = 0
            image_count = len(image_data)
            batch = False

            # setup batch params
            if queue_object.batch[0] > 1 or queue_object.batch[1] > 1:
                batch = True
                grids = []
                images = []
                aspect_ratio = queue_object.width / queue_object.height
                num_grids = math.ceil(image_count / 25)
                grid_count = 25 if num_grids > 1 else image_count
                last_grid_count = image_count % 25
                if num_grids > 1 and image_count % 25 == 0:
                    last_grid_count = 25

                if aspect_ratio <= 1:
                    grid_cols = int(math.ceil(math.sqrt(grid_count)))
                    grid_rows = math.ceil(grid_count / grid_cols)
                    if last_grid_count > 0:
                        last_grid_cols = int(math.ceil(math.sqrt(last_grid_count)))
                        last_grid_rows = math.ceil(last_grid_count / last_grid_cols)
                else:
                    grid_rows = int(math.ceil(math.sqrt(grid_count)))
                    grid_cols = math.ceil(grid_count / grid_rows)
                    if last_grid_count > 0:
                        last_grid_rows = int(math.ceil(math.sqrt(last_grid_count)))
                        last_grid_cols = math.ceil(last_grid_count / last_grid_rows)

                for i in range(num_grids):
                    if i == num_grids:
                        continue
                    
                    if i < num_grids - 1 or last_grid_count == 0:
                        width = grid_cols * queue_object.width
                        height = grid_rows * queue_object.height
                    else: 
                        width = last_grid_cols * queue_object.width
                        height = last_grid_rows * queue_object.height
                    image = Image.new('RGB', (width, height))
                    grids.append(image)

            for i in image_data:
                count += 1
                image = Image.open(io.BytesIO(base64.b64decode(i)))

                # grab png info
                png_payload = {
                    "image": "data:image/png;base64," + i
                }
                png_response = s.post(url=f'{settings.global_var.url}/sdapi/v1/png-info', json=png_payload)

                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("parameters", png_response.json().get("info"))
                str_parameters = png_response.json().get("info")

                file_path = f'{settings.global_var.dir}/{epoch_time}-{queue_object.seed}-{count}.png'

                # if we are using a batch we need to save the files to disk
                if settings.global_var.save_outputs == 'True' or batch == True:
                    image.save(file_path, pnginfo=metadata)
                    print(f'Saved image: {file_path}')

                if batch == True:
                    image_data = (image, file_path, str_parameters)
                    images.append(image_data)

                settings.stats_count(1)

                # increment epoch_time for view when using batch
                if count != len(image_data):
                    new_epoch = list(queue_object.view.input_tuple)
                    new_epoch[18] = int(time.time())
                    new_tuple = tuple(new_epoch)
                    queue_object.view.input_tuple = new_tuple

                #if queue_object.poseref is not None or queue_object.ipadapter is not None:
                #    break

            # progression flag, job done
            queue_object.is_done = True

            # update the leaderboard
            batch_total = queue_object.batch[0] * queue_object.batch[1]
            for _ in range(batch_total):
                LeaderboardCog.update_leaderboard(queue_object.ctx.author.id, str(queue_object.ctx.author), "Image_Count")

            # set up discord message
            content = f'> for {queue_object.ctx.author.name}'
            noun_descriptor = "drawing" if image_count == 1 else f'{image_count} drawings'
            draw_time = '{0:.3f}'.format(end_time - start_time)
            model_name = queue_object.data_model.split('.safetensors')[0]
            message = f'my {noun_descriptor} of ``{queue_object.simple_prompt}`` with ``{model_name}`` took me ``{draw_time}`` seconds!'

            view = queue_object.view

            if batch == True:
                current_grid = 0
                grid_index = 0
                for grid_image in images:
                    if grid_index >= grid_count:
                        grid_index = 0
                        current_grid += 1

                    if current_grid < num_grids - 1 or last_grid_count == 0:
                        grid_y, grid_x = divmod(grid_index, grid_cols)
                        grid_x *= queue_object.width
                        grid_y *= queue_object.height
                    else:
                        grid_y, grid_x = divmod(grid_index, last_grid_cols)
                        grid_x *= queue_object.width
                        grid_y *= queue_object.height

                    grids[current_grid].paste(grid_image[0], (grid_x, grid_y))
                    grid_index += 1

                
                current_grid = 0
                for grid in grids:
                    if current_grid < num_grids -1 or last_grid_count == 0:
                        id_start = current_grid * grid_count + 1
                        id_end = id_start + grid_count - 1
                    else:
                        id_start = current_grid * grid_count + 1
                        id_end = id_start + last_grid_count - 1
                    filename=f'{queue_object.seed}-{current_grid}.png'
                    file = add_metadata_to_image(grid,images[current_grid * 25][2], filename)
                    if current_grid == 0:
                        content = f'<@{queue_object.ctx.author.id}>, {message}\n Batch ID: {epoch_time}-{queue_object.seed}\n Image IDs: {id_start}-{id_end}'
                    else:
                        content = f'> for {queue_object.ctx.author.name}, use /info or context menu to retrieve.\n Batch ID: {epoch_time}-{queue_object.seed}\n Image IDs: {id_start}-{id_end}'
                        view = None
                        
                    current_grid += 1
                    # post discord message
                    queuehandler.process_post(
                        self, queuehandler.PostObject(
                            self, queue_object.ctx, content=content, file=file, embed='', view=view))

            else:
                content = f'<@{queue_object.ctx.author.id}>, {message}'
                # Apply adaptive color correction + sharpening if Details++ is selected
                if getattr(queue_object, "adetailer", None) == 'Details++':
                    # Resize first (optionnel selon workflow)
                    image = image.resize((int(queue_object.width * 0.75), int(queue_object.height * 0.75)))
                    image = apply_color_correction(image)
                filename = f'{queue_object.seed}-{count}.png'
                file = add_metadata_to_image(image, str_parameters, filename)
                queuehandler.process_post(
                    self, queuehandler.PostObject(
                        self, queue_object.ctx, content=content, file=file, embed='', view=view))

        except KeyError as e:
            embed = discord.Embed(title='txt2img failed', description=f'An invalid parameter was found!\nKey causing the error: {e}',
                                color=settings.global_var.embed_color)
            event_loop.create_task(queue_object.ctx.channel.send(embed=embed))
        except Exception as e:
            embed = discord.Embed(title='txt2img failed', description=f'{e}\n{traceback.print_exc()}',
                                  color=settings.global_var.embed_color)
            event_loop.create_task(queue_object.ctx.channel.send(embed=embed))
        
        # check each queue for any remaining tasks
        GlobalQueue.process_queue()


def setup(bot):
    bot.add_cog(StableCog(bot))

def add_metadata_to_image(image, str_parameters, filename):
    with io.BytesIO() as buffer:
        # setup metadata
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("parameters", str_parameters)
        # save image to buffer
        image.save(buffer, 'PNG', pnginfo=metadata)

        # reset buffer to beginning and return as bytes
        buffer.seek(0)
        file = discord.File(fp=buffer, filename=filename)

    return file
