import os, traceback
import asyncio
import aiohttp
import discord
from discord.ext import commands
from discord import option
import requests
import typing, functools
import json
 
 
class DeforumCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        if not os.path.exists('deforum.json'):
            # Si le fichier n'existe pas, créez-le avec le contenu par défaut
            default_content = {
                "URL": "http://127.0.0.1:7860/deforum/run",
                "patreon": "https://patreon.com/deforum",
                "anim_waittime": 10,
                "true_frames_limit": 100
            }
            with open('deforum.json', 'w') as cfg_file:
                json.dump(default_content, cfg_file, indent=4)

        with open('deforum.json', 'r') as cfg_file:
            self.config = json.loads(cfg_file.read())
        assert self.config is not None
 
        # Load the existing configuration or create a new one if it doesn't exist
        self.config = self.load_or_create_config('deforum.json')
    
    def load_default_settings(self, filename):
        """Load the default settings from a file."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Default settings file '{filename}' not found!")
        
        with open(filename, 'r') as file:
            return json.load(file)

    def load_or_create_config(self, filename):
        """Load the existing configuration or create a new one if it doesn't exist."""
        if not os.path.exists(filename):
            default_content = {
                "URL": "http://127.0.0.1:7860/deforum_api/batches",
                "patreon": "https://patreon.com/deforum",
                "anim_waittime": 10,
                "true_frames_limit": 100
            }
            with open(filename, 'w') as cfg_file:
                json.dump(default_content, cfg_file, indent=4)
            return default_content
        else:
            with open(filename, 'r') as cfg_file:
                return json.load(cfg_file)

    def to_thread(func: typing.Callable) -> typing.Coroutine:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.to_thread(func, *args, **kwargs)
        return wrapper
 
    async def make_animation(self, deforum_settings):
        try:
            url = "http://127.0.0.1:7860/deforum_api/batches"
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
                        await asyncio.sleep(10)  # Wait for 10 seconds
                        
                        while True:
                            # Check the job status
                            job_status = await self.check_job_status(session, job_id)
                            
                            if job_status == "SUCCEEDED":
                                # If the job succeeded, get the output directory
                                job_data_url = f"http://127.0.0.1:7860/deforum_api/jobs/{job_id}"
                                async with session.get(job_data_url) as job_data_response:
                                    job_data = await job_data_response.json()
                                    return job_data['outdir']
                                break
                            elif job_status == "FAILED":
                                print(f"Job failed. Status: {job_status}")
                                return ""
                            else:
                                print(f"Job status: {job_status}. Waiting for completion...")
                                await asyncio.sleep(10)  # Wait for 10 seconds before checking again
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
        job_status_url = f"http://127.0.0.1:7860/deforum_api/jobs"
        max_retries = 5
        retries = 0
        while retries < max_retries:
            try:
                async with session.get(job_status_url) as response:
                    data = await response.json()
                    if job_id in data:
                        status = data[job_id]['status']
                        return status
                    else:
                        print(f"Job {job_id} not found. Retrying in 5 seconds...")
                        await asyncio.sleep(5)
                        retries += 1

            except aiohttp.client_exceptions.ClientOSError:
                print("Network error encountered. Retrying in 5 seconds...")
                await asyncio.sleep(5)
                retries += 1

        print(f"Failed to get job status after {max_retries} attempts.")
        return "ERROR"  # ou lever une exception personnalisée

 
    def parse_prompts(self, string, filename='unknown'):
        frames = dict()
        for match_object in string.split(","):
            frameParam = match_object.split(":")
            try:
                frame = frameParam[0].strip()
                frames[frame] = frameParam[1].strip()
            except SyntaxError as e:
                e.filename = filename
                raise e
        if frames == {} and len(string) != 0:
            traceback.print_exc()
            raise RuntimeError('Key Frame string not correctly formatted')
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
 
    def wrap_value(self, val:str):
        val = val.strip()
        if len(val) > 0 and not '(' in val and not ')' in val:
            val = f'0:({val})'
        return val
    
    @commands.slash_command(name='deforum', description='Create an animation based on provided parameters.', guild_only=True)
    @option('prompts', str, description='The text for the animation.', required=True)
    @option('cadence', int, description='The cadence for the animation. (default=6)', required=False, default=6)
    @option('seed', int, description='The seed for the animation. (default=-1)', required=False, default=-1)
    @option('translation_x', str, description='The X translation for the animation. (default="0:(0)"))', required=False, default="0:(0)")
    @option('translation_y', str, description='The Y translation for the animation.(default="0:(0)"))', required=False, default="0:(0)")
    @option('translation_z', str, description='The Z translation (zoom) for the animation. (default="0:(1.5)"))', required=False, default="0:(1.5)")
    @option('rotation_3d_x', str, description='The 3D X rotation (around horizontal axe) for the animation. (default="0:(0)"))', required=False, default="0:(0)")
    @option('rotation_3d_y', str, description='The 3D Y rotation (around vertical axe)for the animation. (default="0:(0)"))', required=False, default="0:(0)")
    @option('rotation_3d_z', str, description='The 3D Z rotation (angle rotation) for the animation. (default="0:(0)"))', required=False, default="0:(0)")
    @option('width', int, description='The width for the animation. (default=768))', required=False, default=768)
    @option('height', int, description='The height for the animation. (default=512))', required=False, default=512)
    @option('fps', int, description='The fps for the animation. (default=15))', required=False, default=15)
    @option('max_frames', int, description='The total frames for the animation. (default=120))', required=False, default=120)
    @option('fov_schedule', str, description='Adjust the FOV. (default="0:(120)"))', required=False, default="0:(120)")
    @option('noise_schedule', str, description='Adjust the Noise Schedule. (default="0:(0.065)"))', required=False, default="0:(0.065)")
    @option('noise_multiplier_schedule', str, description='Adjust the Noise Multiplier Schedule. (default="0:(1.05)"))', required=False, default="0:(1.05)")
    @option('strength_schedule', str, description='Adjust the Strength Schedule. (default="0:(0.65)"))', required=False, default="0:(0.65)")
    @option('cfg_scale_schedule', str, description='Adjust the CFG Scale Schedule. (default="0:(7)"))', required=False, default="0:(7)")
    @option('antiblur_amount_schedule', str, description='Adjust the Anti-Blur Amount Schedule. (default="0:(0.1)"))', required=False, default="0:(0.1)")
    @option('parseq_manifest', str, description='Parseq Manifest URL to use', required=False, default="")
    #@option('add_soundtrack', str, description='Soundtrack option', required=False, default="")
    
    async def deforum_handler(
        self,
        ctx,
        prompts: str = "",
        cadence: int = 6,
        steps: int = 25,
        seed: int = -1,
        strength: str = "0:(0.65)",
        translation_x: str = "0:(0)",
        translation_y: str = "0:(0)",
        translation_z: str = "0:(1.5)",
        rotation_3d_x: str = "0:(0)",
        rotation_3d_y: str = "0:(0)",
        rotation_3d_z: str = "0:(0)",
        width: int = 768,
        height: int = 512,
        fps: int = 15,
        max_frames: int = 120,
        fov_schedule: str = "0:(120)",
        noise_schedule: str = "0:(0.065)",
        noise_multiplier_schedule: str = "0:(1.05)",
        strength_schedule: str = "0:(0.65)",
        cfg_scale_schedule: str = "0:(7)",
        antiblur_amount_schedule: str = "0:(0.1)",
        parseq_manifest: str = ""
    ):
        print(f'/Deforum request -- {ctx.author.name} -- Seed: {seed} Prompts: {prompts}\nCadence: {cadence}, strength: {strength}, Width: {width}, Height: {height}, FPS:{fps}, Seed:{seed}, Max Frames: {max_frames}')

        # Load the default settings from the file
        with open('default_settings.json', 'r') as settings_file:
            deforum_settings = json.load(settings_file)

        if cadence <= 2 or width  > 1216 or height  > 1216 or fps < 1 or max_frames < 1:
            await ctx.respond(f"<@{ctx.author.id}> Cadence must be greater than 2, width and height can't be larger than 1024 and fps not less than 1")
            return
        if max_frames / cadence > self.config['true_frames_limit']:
            await ctx.respond(f"<@{ctx.author.id}> With Cadence {cadence} the length of the animation is limited by {cadence * self.config['true_frames_limit']} frames")
            return
        
        
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

        # addfixed params to deforum_settings
        deforum_settings['sampler'] = "DPM++ 2M Karras"
        deforum_settings['motion_preview_mode'] = False # changed to fixeds for now
        deforum_settings['animation_mode'] = "3D"
        deforum_settings['border'] = "wrap"
        deforum_settings['padding_mode'] = "reflection"

        # add useful options to deforum_settings
        deforum_settings['diffusion_cadence'] = cadence
        deforum_settings['steps'] = steps
        deforum_settings['W'] = width
        deforum_settings['H'] = height
        deforum_settings['fps'] = fps
        deforum_settings['seed'] = seed
        deforum_settings['max_frames'] = max_frames
        deforum_settings['parseq_manifest'] = parseq_manifest

        print('Parsing prompts')
        try:
            prompts = self.parse_prompts(prompts)
        except Exception as e:
            await ctx.respond(f'<@{ctx.author.id}> Error parsing prompts!')
            return
        
        deforum_settings["prompts"] = prompts 
        #print(f'deforum_settings: {deforum_settings}')

        await ctx.respond(f'<@{ctx.author.id}> Making a Deforum animation...')
        print('Making a Deforum animation...')
        path = await self.make_animation(deforum_settings)

        if len(path) > 0:
            print('Animation made.')
            anim_file = self.find_animation(os.path.abspath(path))
            await ctx.send(file=discord.File(anim_file))
            settings_file = self.find_settings(os.path.abspath(path))
            result_seed = -2
            try:
                with open(settings_file, 'r', encoding='utf-8') as sttn:
                    result_settings = json.loads(sttn.read())
                result_seed = result_settings['seed']
            except:
                ...
            await ctx.send(file=discord.File(settings_file)) # feature for selected users?
            await ctx.respond((f'<@{ctx.author.id}> Your animation is done!') + (f' Seed used: {result_seed}' if result_seed != -2 else ''))
            #await ctx.respond((f'<@{ctx.author.id}> Your animation is done!' if not motion_preview_mode else 'Your movement preview is done!') + (f' Seed used: {result_seed}' if result_seed != -2 else ''))

        else:
            await ctx.respond(f'<@{ctx.author.id}> Sorry, there was an error making the animation!')
 
def setup(bot):
    bot.add_cog(DeforumCog(bot))
 