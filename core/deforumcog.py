import os, traceback
import asyncio
from asyncio import AbstractEventLoop
from asyncio import run_coroutine_threadsafe
import aiohttp
import discord
from discord import option
from discord.ext import commands
from discord.ui import Button, View
import typing, functools
from typing import Optional
import json
import random
import re
import requests

from core import settings
from core import queuehandler
from core import viewhandler
from core.queuehandler import GlobalQueue
from core.leaderboardcog import LeaderboardCog


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


class DeforumView(View):
    def __init__(self, ctx, message, queue_object):
        super().__init__(timeout=None)
        self.ctx = ctx
        self.message = message
        self.deforum_settings = queue_object.deforum_settings

        self.add_item(DeleteButton(self))


class DeforumCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config = self.load_or_create_config('deforum.json')

    def load_or_create_config(self, filename):
        """Load the existing configuration or create a new one if it doesn't exist."""
        if not os.path.exists(filename):
            default_content = {
                "URL": "http://127.0.0.1:7860/deforum_api",
                "patreon": "https://patreon.com/deforum",
                "true_frames_limit": 1000
            }
            with open(filename, 'w') as cfg_file:
                json.dump(default_content, cfg_file, indent=4)
            return default_content
        else:
            with open(filename, 'r') as cfg_file:
                return json.load(cfg_file)

    def load_default_settings(self, filename):
        """Load the default settings from a file."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Default settings file '{filename}' not found!")

        with open(filename, 'r') as file:
            return json.load(file)

    def to_thread(func: typing.Callable) -> typing.Coroutine:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.to_thread(func, *args, **kwargs)
        return wrapper

    async def make_animation(self, deforum_settings):
        try:
            url = f"{self.config['URL']}/batches"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={
                    "deforum_settings": [deforum_settings],
                    "options_overrides": {
                        "deforum_save_gen_info_as_srt": False,
                    }
                }) as response:
                    # Check if the response status is 202 (Accepted)
                    if response.status == 202:
                        print("Job has been accepted for processing.")

                        # Extract job_id from the response
                        data = await response.json()
                        job_id = data["job_ids"][0]
                        print(f"Received job_id: {job_id}")

                        # Wait for a short duration before checking the job status
                        await asyncio.sleep(10)

                        while True:
                            # Check the job status
                            job_status = await self.check_job_status(session, job_id)                            
                            if job_status == "SUCCEEDED":
                                # If the job succeeded, get the output directory
                                job_data_url = f"{self.config['URL']}/jobs/{job_id}"
                                async with session.get(job_data_url) as job_data_response:
                                    job_data = await job_data_response.json()
                                    return job_data['outdir']
                                break
                            elif job_status == "FAILED":
                                print(f"Job failed. Status: {job_status}")
                                return ""
                            else:
                                #print(f"Job status: {job_status}. Waiting for completion...")
                                await asyncio.sleep(2)
                    else:
                        print(f"Bad status code {response.status}")
                        if response.status == 422:
                            details = await response.json()
                            print(details['detail'])
                        return ""
        except Exception as e:
            traceback.print_exc()
            print(e)
            return ""

    async def check_job_status(self, session, job_id):
        job_status_url = f"{self.config['URL']}/jobs"
        max_retries = 30
        retries = 0
        while retries < max_retries:
            try:
                async with session.get(job_status_url) as response:
                    data = await response.json()
                    if job_id in data:
                        status = data[job_id]['status']
                        return status
                    else:
                        print(f"Job {job_id} not found. Retrying in 1 second...")
                        await asyncio.sleep(1)
                        retries += 1

            except aiohttp.client_exceptions.ClientOSError:
                #print("Network error encountered. Retrying in 1 seconds...")
                await asyncio.sleep(1)
                retries += 1

        print(f"Failed to get job status after {max_retries} seconds.")
        return "ERROR"

    def parse_prompts(self, string, filename='unknown'):
        frames = dict()
        stack = []
        start_idx = None
        frame_num = None

        i = 0
        while i < len(string):
            char = string[i]
            if char == "(":
                stack.append(char)
                if len(stack) == 1:
                    # new content
                    start_idx = i + 1
                    # extract the frame number
                    frame_num_match = re.search(r'(\d+)\s*:', string[max(0, i-15):i])
                    if frame_num_match:
                        frame_num = frame_num_match.group(1)
            elif char == ")":
                stack.pop()
                if not stack:
                    content = string[start_idx:i].strip()
                    if frame_num:
                        frames[frame_num] = content
                        frame_num = None
            i += 1

        # if the string was not parsed (no parentheses detected), assume it's for frame 0
        if not frames:
            frames['0'] = string.strip()

        return frames

    def find_animation(self, d):
        for f in os.listdir(d):
            if f.endswith('.mp4'):
                return os.path.join(d, f)
        return ''

    def find_settings(self, d):
        for f in os.listdir(d):
            if f.endswith('.txt'):
                return os.path.join(d, f)
        return ''
    
    def find_gif(self, d):
        for f in os.listdir(d):
            if f.endswith('.gif'):
                return os.path.join(d, f)
        return ''

    def wrap_value(self, val:str):
        val = val.strip()
        if len(val) > 0 and not '(' in val and not ')' in val:
            val = f'0:({val})'
        return val

    @commands.slash_command(name='deforum', description='Create an animation based on provided parameters.', guild_only=True)
    @option('prompts', str, description='The prompts to generate the animation.', required=True,)
    @option('cadence', int, description='The cadence for the animation. (default=6)', required=False, default=6)
    @option('steps', int, description='The steps for the animation. (default=30)', required=False, default=30)
    @option('seed', int, description='The seed for the animation. (default= -1 = random, iter every frame)', required=False, default=-1)
    @option('size_ratio', str, description='The size ratio for the animation.', required=False, default=None, choices=[
        "Fullscreen: 4:3 - 1152x896",
        "Widescreen: 16:9 - 1344x768",
        "Ultrawide: 21:9 - 1536x640",
        "Landscape: 3:2 - 1280x768",
        "Square: 1:1 - 1024x1024",
        "Portrait: 2:3 - 768x1280",
        "Tall: 9:16 - 768x1344"
    ])
    @option('translation_x', str, description='The X translation to move the canvas left/right in pixels per frame. (default="0:(0)"))', required=False, default="0:(0)")
    @option('translation_y', str, description='The Y translation to move the canvas up/down in pixels per frame.(default="0:(0)"))', required=False, default="0:(0)")
    @option('translation_z', str, description='The Z translation to move the canvas towards/away. [speed based on fov].(default="0:(0.5)"))', required=False, default="0:(0.5)")
    @option('rotation_3d_x', str, description='The 3D X rotation to tilt the canvas up/down in degrees per frame. (default="0:(0)"))', required=False, default="0:(0)")
    @option('rotation_3d_y', str, description='The 3D Y rotation to pan the canvas left/right in degrees per frame. (default="0:(0)"))', required=False, default="0:(0)")
    @option('rotation_3d_z', str, description='The 3D Z rotation to roll the canvas clockwise/ anticlockwise. (default="0:(0)"))', required=False, default="0:(0)")
    @option('width', int, description='The width for the animation. (default=768))', required=False, default=768)
    @option('height', int, description='The height for the animation. (default=1280))', required=False, default=1280)
    @option('fps', int, description='The FPS for the animation. (default=15))', required=False, default=15)
    @option('max_frames', int, description='The total frames for the animation. (default=10))', required=False, default=120)
    @option('fov_schedule', str, description='Adjust the FOV. (default="0:(140)"))', required=False, default="0:(140)")
    @option('noise_schedule', str, description='Adjust the Noise Schedule. (default="0:(0)"))', required=False, default="0:(0)")
    @option('noise_multiplier_schedule', str, description='Adjust the Noise Multiplier Schedule. (default="0:(1.06)"))', required=False, default="0:(1.06)")
    @option('strength_schedule', str, description='Adjust the Strength Schedule. (default="0:(0.7)"))', required=False, default="0:(0.7)")
    @option('cfg_scale_schedule', str, description='Adjust the CFG Scale Schedule. (default="0:(9)"))', required=False, default="0:(9)")
    @option('antiblur_amount_schedule', str, description='Adjust the Anti-Blur Amount Schedule. (default="0:(0.25)"))', required=False, default="0:(0.25)")
    #@option('add_soundtrack', discord.Attachment, description="Attach a soundtrack MP3 file. It doesn't need to match the video duration.", required=False)
    @option('frame_interpolation_engine', str, description='Enable the frame interpolation engine. Triples video generation time. (default="None", x3 FPS)', required=False, default="None", choices=["None", "FILM"])
    @option('parseq_manifest', str, description='Parseq Manifest URL to use. Fields managed by Parseq override the values set in other options.', required=False, default="")
    @option('init_image',str,description='The starter URL image for generation.', required=False)
    @option('make_gif', bool, description='Produce a GIF version of the animation.', required=False, default=False)

    async def deforum_handler(
        self,
        ctx,
        prompts: str,
        cadence: Optional[int] = 5,
        steps: Optional[int] = 30,
        seed: Optional[int] = -1,
        size_ratio: Optional[str] = None,
        translation_x: Optional[str] = "0:(0)",
        translation_y: Optional[str] = "0:(0)",
        translation_z: Optional[str] = "0:(0.5)",
        rotation_3d_x: Optional[str] = "0:(0)",
        rotation_3d_y: Optional[str] = "0:(0)",
        rotation_3d_z: Optional[str] = "0:(0)",
        width: Optional[int] = 768,
        height: Optional[int] = 1280,
        fps: Optional[int] = 15,
        max_frames: Optional[int] = 120,
        fov_schedule: Optional[str] = "0:(140)",
        noise_schedule: Optional[str] = "0:(0)",
        noise_multiplier_schedule: Optional[str] = "0:(1.06)",
        strength_schedule: Optional[str] = "0:(0.7)",
        cfg_scale_schedule: Optional[str] = "0:(9)",
        antiblur_amount_schedule: Optional[str] = "0:(0.25)",
        #add_soundtrack: discord.Attachment = None,
        frame_interpolation_engine: Optional[str] = "None",
        parseq_manifest: Optional[str] = "",
        init_image: Optional[str] = "",
        make_gif: Optional[bool] = False
    ):

        # size ratioss
        ratios = {
            "Fullscreen: 4:3 - 1152x896": (1152, 896),
            "Widescreen: 16:9 - 1344x768": (1344, 768),
            "Ultrawide: 21:9 - 1536x640": (1536, 640),
            "Landscape: 3:2 - 1280x768": (1280, 768),
            "Square: 1:1 - 1024x1024": (1024, 1024),
            "Portrait: 2:3 - 768x1280": (768, 1280),
            "Tall: 9:16 - 768x1344": (768, 1344)
        }

        # ratio override width and height if size_ratio is provided
        if size_ratio and size_ratio in ratios:
            width, height = ratios[size_ratio]

        print(f'/Deforum request -- {ctx.author.name} -- Seed: {seed} Prompts: {prompts}\nCadence: {cadence}, Width: {width}, Height: {height}, FPS:{fps}, Seed:{seed}, Max Frames: {max_frames}')

        # construct a dic for the default parameters to compare
        default_params = {
            "prompts": "",
            "cadence": 6,
            "steps": 30,
            "seed": "",
            "translation_x": "0:(0)",
            "translation_y": "0:(0)",
            "translation_z": "0:(0.5)",
            "rotation_3d_x": "0:(0)",
            "rotation_3d_y": "0:(0)",
            "rotation_3d_z": "0:(0)",
            "width": 768,
            "height": 1280,
            "fps": 15,
            "max_frames": 120,
            "fov_schedule": "0:(140)",
            "noise_schedule": "0:(0)",
            "noise_multiplier_schedule": "0:(1.06)",
            "strength_schedule": "0:(0.7)",
            "cfg_scale_schedule": "0:(9)",
            "antiblur_amount_schedule": "0:(0.25)",
            "frame_interpolation_engine": "None",
            #"add_soundtrack": discord.Attachment = None,
            "parseq_manifest": "",
            "init_image": "",
            "make_gif": False
        }

        # mapping dic for display names 
        key_mapping = {
            "prompts":  "Prompts",
            "cadence": "Cadence",
            "steps": "Steps",
            "seed": "Seed",
            "translation_x": "Translation X",
            "translation_y": "Translation Y",
            "translation_z": "Translation Z",
            "rotation_3d_x": "Rotation X",
            "rotation_3d_y": "Rotation Y",
            "rotation_3d_z": "Rotation Z",
            "width": "Width",
            "height": "Height",
            "fps": "FPS",
            "max_frames": "Total Frames",
            "fov_schedule": "FOV Schedule",
            "noise_schedule": "Noise Schedule",
            "noise_multiplier_schedule": "Noise Multiplier Schedule",
            "strength_schedule": "Strength Schedule",
            "cfg_scale_schedule": "CFG Scale Schedule",
            "antiblur_amount_schedule": "Antiblur Amount Schedule",
            "frame_interpolation_engine": "Frame Interpolation Engine",
            #"add_soundtrack": discord.Attachment = None,
            "parseq_manifest": "Parseq Manifest",
            "init_image": "Init Image",
            "make_gif": "Make GIF"
        }

        # load the default settings from the file
        with open('default_settings.json', 'r') as settings_file:
            deforum_settings = json.load(settings_file)

        if cadence < 1 or width  > 1536 or height  > 1536 or fps < 1 or max_frames < 1:
            await ctx.respond(f"<@{ctx.author.id}> Cadence must be greater than 0, width and height can't be larger than 1536 max frames greater than 1 and fps not less than 1")
            return
        if max_frames / cadence > self.config['true_frames_limit']:
            await ctx.respond(f"<@{ctx.author.id}> With Cadence {cadence} the length of the animation is limited by {cadence * self.config['true_frames_limit']} frames")
            return

        #if add_soundtrack:
            # Now you can download or process the add_soundtrack as needed
            #mp3_file = await add_soundtrack.to_file()
            # Process the mp3_file...

        # construct deforum_settings
        # add wrapped values to deforum_settings
        deforum_settings['translation_x'] = self.wrap_value(translation_x)        
        deforum_settings['translation_y'] = self.wrap_value(translation_y)
        deforum_settings['translation_z'] = self.wrap_value(translation_z)
        deforum_settings['rotation_3d_x'] = self.wrap_value(rotation_3d_x)
        deforum_settings['rotation_3d_y'] = self.wrap_value(rotation_3d_y)
        deforum_settings['rotation_3d_z'] = self.wrap_value(rotation_3d_z)
        deforum_settings['fov_schedule'] = self.wrap_value(fov_schedule)
        deforum_settings['noise_schedule'] = self.wrap_value(noise_schedule)
        deforum_settings['noise_multiplier_schedule'] = self.wrap_value(noise_multiplier_schedule)
        deforum_settings['strength_schedule'] = self.wrap_value(strength_schedule)
        deforum_settings['cfg_scale_schedule'] = self.wrap_value(cfg_scale_schedule)
        deforum_settings['amount_schedule'] = self.wrap_value(antiblur_amount_schedule)

        # add fixed params to deforum_settings
        deforum_settings['sampler'] = "Euler a"
        deforum_settings['motion_preview_mode'] = False # fixed for now
        deforum_settings['animation_mode'] = "3D"        
        deforum_settings['depth_algorithm'] = "Leres"
        deforum_settings['border'] = "wrap"
        deforum_settings['padding_mode'] = "reflection"

        # add useful options to deforum_settings
        deforum_settings['diffusion_cadence'] = cadence
        deforum_settings['steps'] = steps
        deforum_settings['W'] = width
        deforum_settings['H'] = height
        deforum_settings['fps'] = fps

        # randomize the seed if still -1, 10 digits
        if seed == -1:
            seed = random.randint(1000000000, 9999999999)
        deforum_settings['seed'] = seed
        deforum_settings['max_frames'] = max_frames
        deforum_settings['frame_interpolation_engine'] = frame_interpolation_engine
        deforum_settings['parseq_manifest'] = parseq_manifest
        deforum_settings['make_gif'] = make_gif

        # enable and add init_image url to deforum_settings
        if init_image:
            deforum_settings['use_init'] = True
            deforum_settings['strength'] = 0.9
        deforum_settings['init_image'] = init_image
        print(f'init_image: {init_image}')

        try:
            prompts = self.parse_prompts(prompts)
        except Exception as e:
            await ctx.respond(f'<@{ctx.author.id}> Error parsing prompts!')
            return
        deforum_settings["prompts"] = prompts 

        # set up tuple of parameters to pass into the Discord view
        input_tuple = (ctx, deforum_settings)
        view = viewhandler.DeleteView(input_tuple)

        # set up the queue if an image was found
        user_queue_limit = settings.queue_check(ctx.author)

        if queuehandler.GlobalQueue.dream_thread.is_alive():
            if user_queue_limit == "Stop":
                await ctx.send_response(content=f"Please wait! You're past your queue limit of {settings.global_var.queue_limit}.", ephemeral=True)
            else:
                queuehandler.GlobalQueue.queue.append(queuehandler.DeforumObject(self, *input_tuple, view))
        else:
            await queuehandler.process_dream(self, queuehandler.DeforumObject(self, *input_tuple, view))

        # function to compare parameters to default using the key_mapping dic
        def get_display_name(internal_key):
            return key_mapping.get(internal_key, internal_key)

        non_default_params = {get_display_name(key): value for key, value in locals().items() if key in default_params and default_params[key] != value}

        # build the message to send if the options are not default
        message = f'<@{ctx.author.id}>, {settings.messages_deforum()}\nQueue: ``{len(queuehandler.GlobalQueue.queue)}``'
        if non_default_params:
            message += f" - `{non_default_params['Prompts']}`\n"
            del non_default_params['Prompts']
            params_list = list(non_default_params.items())
            for i in range(0, len(params_list), 3):
                line_params = params_list[i:i+3]
                for key, value in line_params:
                    message += f"{key}: `{value}`  "
                message += "\n"
            if parseq_manifest != "":
                message += f"**Parseq Manifest**: `Yes`\n"

        if user_queue_limit != "Stop":
            await ctx.respond(message)

    # the function to queue Discord posts
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

    def dream(self, event_loop: queuehandler.GlobalQueue.event_loop, queue_object: queuehandler.DeforumObject):
        try:
            # start progression message
            run_coroutine_threadsafe(GlobalQueue.update_progress_message(queue_object), event_loop)

            print('Making a Deforum animation...')
            deforum_settings = queue_object.deforum_settings

            # run generation
            future = run_coroutine_threadsafe(self.make_animation(deforum_settings), event_loop)
            path = future.result()

            # schedule the task to create the view and send the message
            event_loop.create_task(self.post_dream(queue_object.ctx, queue_object, path))

            # progression flag, job done
            queue_object.is_done = True

        except Exception as e:
            embed = discord.Embed(title='Generation failed', description=f'{e}\n{traceback.print_exc()}', color=0x00ff00)
            event_loop.create_task(queue_object.ctx.channel.send(embed=embed))

        # update the leaderboard
        LeaderboardCog.update_leaderboard(queue_object.ctx.author.id, str(queue_object.ctx.author), "Deforum_Count")

        # check each queue for any remaining tasks
        GlobalQueue.process_queue()

    # post to discord
    async def post_dream(self, ctx, queue_object, path):
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB in bytes

        def get_file_size_str(file_path):
            size_bytes = os.path.getsize(file_path)
            size_mb = size_bytes / (1024 * 1024)
            return f"{size_mb:.2f} MB"

        if len(path) > 0:
            print('Animation made.')
            anim_file = self.find_animation(os.path.abspath(path))
            settings_file = self.find_settings(os.path.abspath(path))
            gif_file = self.find_gif(os.path.abspath(path))
            result_seed = -1
            try:
                with open(settings_file, 'r', encoding='utf-8') as sttn:
                    result_settings = json.loads(sttn.read())
                result_seed = result_settings['seed']
            except:
                ...

            # create view
            view = DeforumView(ctx, None, queue_object)

            # check file sizes and possibly exclude large files...
            if os.path.getsize(anim_file) > MAX_FILE_SIZE:
                file_size_str = get_file_size_str(anim_file)
                await ctx.send(f"<@{ctx.author.id}> The animation file size ({file_size_str}) exceeds Discord's 50MB limit and can't be uploaded...")
                anim_file = None

            if gif_file and os.path.getsize(gif_file) > MAX_FILE_SIZE:
                file_size_str = get_file_size_str(gif_file)
                await ctx.send(f"<@{ctx.author.id}> The GIF file size ({file_size_str}) exceeds Discord's 50MB limit and can't be uploaded...")
                gif_file = None

            # send the animation (if it's under the limit) with the view attached
            if anim_file:
                message = await ctx.send(f'<@{ctx.author.id}>, {settings.messages_deforum_end()}\nSeed used: {result_seed}', file=discord.File(anim_file), view=view)
                view.message = message

            # send the video settings. If make_gif is true and gif is under limit, attach the GIF
            files_to_send = [discord.File(settings_file)]
            if gif_file and queue_object.deforum_settings.get('make_gif'):
                files_to_send.append(discord.File(gif_file))

            # adjust the message
            if len(files_to_send) > 1:
                file_message = f'<@{ctx.author.id}> Additional files for the Animation with the Seed: {result_seed}'
            else:
                file_message = f'<@{ctx.author.id}> Additional file for the Animation with the Seed: {result_seed}'

            # add the view to this message as well
            await ctx.send(file_message, files=files_to_send, view=view)

        else:
            await ctx.respond(f'<@{queue_object.ctx.author.id}> Sorry, there was an error making the animation!')


def setup(bot):
    bot.add_cog(DeforumCog(bot))
 