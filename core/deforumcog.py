import os, traceback
import asyncio
import discord
import time
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
 
    def to_thread(func: typing.Callable) -> typing.Coroutine:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.to_thread(func, *args, **kwargs)
        return wrapper
 
    @to_thread
    def make_animation(self, deforum_settings):
        global config
        settings_json = json.dumps(deforum_settings)
        allowed_params = ';'.join(deforum_settings.keys()).strip(';')
        request = {'settings_json':settings_json, 'allowed_params':allowed_params}
        try:
            response = requests.post(self.config['URL'], params=request)
            if response.status_code == 200:
                result = response.json()['outdir']
            else:
                print(f"Bad status code {response.status_code}")
                if response.status_code == 422:
                    print(response.json()['detail'])
                return ""
        except Exception as e:
            traceback.print_exc()
            print(e)
            return ""
        return result
 
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
            val = f'0: ({val})'
        return val
    
    @commands.slash_command(name='deforum', description='Create an animation based on provided parameters.', guild_only=True)
    @option('prompts', str, description='The text for the animation.', required=True)
    @option('cadence', int, description='The cadence for the animation. (default=6)', required=False, default=6)
    @option('seed', int, description='The seed for the animation. (default=-1)', required=False, default=-1)
    @option('strength', str, description='The strength for the animation.(default="0: (0.65)"))', required=False, default="0: (0.65)")
    @option('speed_x', str, description='The speed x for the animation. (default="0: (0)"))', required=False, default="0: (0)")
    @option('speed_y', str, description='The speed y for the animation.(default="0: (0)"))', required=False, default="0: (0)")
    @option('speed_z', str, description='The speed z for the animation. (default="0: (1.5)"))', required=False, default="0: (1.5)")
    @option('rotate_x', str, description='The rotate x for the animation. (default="0: (0)"))', required=False, default="0: (0)")
    @option('rotate_y', str, description='The rotate y for the animation. (default="0: (0)"))', required=False, default="0: (0)")
    @option('rotate_z', str, description='The rotate z for the animation. (default="0: (0)"))', required=False, default="0: (0)")
    @option('width', int, description='The width for the animation. (default=768))', required=False, default=768)
    @option('height', int, description='The height for the animation. (default=512))', required=False, default=512)
    @option('fps', int, description='The fps for the animation. (default=15))', required=False, default=15)
    @option('max_frames', int, description='The total frames for the animation. (default=120))', required=False, default=120)
    @option('fov_schedule', str, description='Adjust the FOV. (default="0: (120)"))', required=False, default="0: (120)")
    @option('noise_schedule', str, description='Adjust the Noise Schedule. (default="0: (0.065)"))', required=False, default="0: (0.065)")
    @option('noise_multiplier_schedule', str, description='Adjust the Noise Multiplier Schedule. (default="0: (1.05)"))', required=False, default="0: (1.05)")
    @option('strength_schedule', str, description='Adjust the Strength Schedule. (default="0: (0.65)"))', required=False, default="0: (0.65)")
    @option('cfg_scale_schedule', str, description='Adjust the CFG Scale Schedule. (default="0: (7)"))', required=False, default="0: (7)")
    @option('antiblur_amount_schedule', str, description='Adjust the Anti-Blur Amount Schedule. (default="0: (0.1)"))', required=False, default="0: (0.1)")
    @option('parseq_manifest', str, description='Parseq Manifest URL to use', required=False, default="")

    async def deforum_handler(
        self, 
        ctx, 
        prompts: str = "", 
        cadence: int = 6, 
        steps: int = 25,
        seed: int = -1, 
        strength: str = "0: (0.65)", 
        speed_x: str = "0: (0)", 
        speed_y: str = "0: (0)", 
        speed_z: str = "0: (1.5)", 
        rotate_x: str = "0: (0)", 
        rotate_y: str = "0: (0)", 
        rotate_z: str = "0: (0)", 
        width: int = 768, 
        height: int = 512, 
        fps: int = 15, 
        max_frames: int = 120,
        fov_schedule: int = "0: (120)",
        noise_schedule: str = "0: (0.065)",
        noise_multiplier_schedule: str = "0: (1.05)",
        strength_schedule: str = "0: (0.65)",
        cfg_scale_schedule: str = "0: (7)",
        antiblur_amount_schedule: str = "0: (0.1)",
        parseq_manifest: str = ""
    ):
        print(f'/Deforum request -- {ctx.author.name} -- Seed: {seed} Prompts: {prompts}\nCadence: {cadence}, strength: {strength}, Width: {width}, Height: {height}, FPS:{fps}, Seed:{seed}, Max Frames: {max_frames}')

        if cadence < 1 or width  > 1216 or height  > 1216 or fps < 1 or max_frames < 1:
            await ctx.respond("Cadence must be greater than 3, width and height can't be larger than 1024 and fps not less than 1")
            return
        if max_frames / cadence > self.config['true_frames_limit']:
            await ctx.respond(f"With Cadence {cadence} the length of the animation is limited by {cadence * self.config['true_frames_limit']} frames")
            return
        prompts = self.wrap_value(prompts)
        strength = self.wrap_value(strength)
        speed_x = self.wrap_value(speed_x)        
        speed_y = self.wrap_value(speed_y)
        speed_z = self.wrap_value(speed_z)
        rotate_x = self.wrap_value(rotate_x)
        rotate_y = self.wrap_value(rotate_y)
        rotate_z = self.wrap_value(rotate_z)
        fov_schedule = self.wrap_value(fov_schedule)
        noise_schedule = self.wrap_value(noise_schedule)
        noise_multiplier_schedule = self.wrap_value(noise_multiplier_schedule)
        strength_schedule = self.wrap_value(strength_schedule)
        cfg_scale_schedule = self.wrap_value(cfg_scale_schedule)
        amount_schedule = self.wrap_value(antiblur_amount_schedule)

        # fixed params
        sampler = "DPM++ 2M Karras"
        motion_preview_mode = "False" # changed to fixeds for now
        animation_mode = "3D"
        border = "wrap"
        padding_mode = "reflection"

        # add useful options to deforum_settings
        deforum_settings = {
            'diffusion_cadence':cadence, 
            'steps':steps, 
            'W':width, 
            'H':height, 
            'fps':fps, 
            'seed':seed, 
            'max_frames':max_frames, 
            'strength_schedule':strength, 
            'translation_x':speed_x, 
            'translation_y':speed_y, 
            'translation_z':speed_z, 
            'rotation_3d_x':rotate_x, 
            'rotation_3d_y':rotate_y, 
            'rotation_3d_z':rotate_z, 
            'fov_schedule':fov_schedule, 
            'noise_schedule':noise_schedule,
            'noise_multiplier_schedule':noise_multiplier_schedule,
            'strength_schedule':strength_schedule,
            'cfg_scale_schedule':cfg_scale_schedule,
            'amount_schedule':amount_schedule,
            'parseq_manifest':parseq_manifest
        }
        
        # add fixed params to deforum_settings
        deforum_settings = {
            'motion_preview_mode':motion_preview_mode, 
            'animation_mode':animation_mode, 
            'border':border,
            'sampler':sampler,
            'padding_mode':padding_mode
        }

        print('Parsing prompts')
        try:
            prompts = self.parse_prompts(prompts)
        except Exception as e:
            await ctx.respond('Error parsing prompts!')
            return
        
        deforum_settings['prompts'] = prompts 
        #print(f'deforum_settings: {deforum_settings}')

        await ctx.respond('Making a Deforum animation...')

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
            await ctx.respond(('Your animation is done!' if not motion_preview_mode else 'Your movement preview is done!') + (f' Seed used: {result_seed}' if result_seed != -2 else ''))
        else:
            await ctx.respond('Sorry, there was an error making the animation!')
 
def setup(bot):
    bot.add_cog(DeforumCog(bot))
 
