# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import os, traceback, shutil

import torch
from tqdm import tqdm

# Literally 1984
from transformers import AutoFeatureExtractor

import concurrent.futures
import asyncio
import time

import discord
from discord.ext import commands
from discord import option

import requests
import typing, functools
import json

from typing import Optional

with open('deforum.json', 'r') as cfg_file:
    config = json.loads(cfg_file.read())
assert config is not None


class DeforumCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.slash_command(name='deforum', description='Generates a video from a prompt', guild_only=True)
    @option(
        'Prompts',
        discord.Attachment,
        description='Your prompts to produce the video.',
        required=True,
    )
    @option(
        'Cadence',
        discord.Attachment,
        description='...',
        required=False,
    )
    @option(
        'Seed',
        discord.Attachment,
        description='The seed to use for reproducibility.',
        required=False,
    )
    @option(
        'Strength',
        discord.Attachment,
        description='Control how much the presence of the previous frame influences the next frame.',
        required=False,
    )
    @option(
        'Preview Mode',
        discord.Attachment,
        description='Enable the preview mode',
        required=False,
    )
    @option(
        'Speed X',
        discord.Attachment,
        description='...',
        required=False,
    )
    @option(
        'Speed Y',
        discord.Attachment,
        description='...',
        required=False,
    )
    @option(
        'Speed Z',
        discord.Attachment,
        description='...',
        required=False,
    )
    @option(
        'Rotate X',
        discord.Attachment,
        description='Tilt the canvas up/down in degrees per frame.',
        required=False,
    )
    @option(
        'Rotate Y',
        discord.Attachment,
        description='Pan the canvas left/right in degrees per frame.',
        required=False,
    )
    @option(
        'Rotate Z',
        discord.Attachment,
        description='Roll the canvas clockwise/ anticlockwise.',
        required=False,
    )
    @option(
        'Width',
        discord.Attachment,
        description='Width of the generated video.',
        required=False,
    )
    @option(
        'Height',
        discord.Attachment,
        description='Height of the generated video.',
        required=False,
    )
    @option(
        'FPS',
        discord.Attachment,
        description='Frames per second.',
        required=False,
    )
    @option(
        'Total Frames',
        discord.Attachment,
        description='Specifies the number of images to generate in total',
        required=False,
    )    
    async def deforum_handler(self, ctx: discord.ApplicationContext, *,
                            prompts: str, 
                            cadence: int = 6, 
                            seed = -1, 
                            strength: str = "0: (0.65)", 
                            preview_mode: bool = False, 
                            speed_x: str = "0: (0)", 
                            speed_y: str = "0: (0)", 
                            speed_z: str = "0: (1.75)", 
                            rotate_x:str = "0: (0)", 
                            rotate_y: str = "0: (0)", 
                            rotate_z: str = "0: (0)", 
                            width:int = 768, 
                            height: int = 768, 
                            fps: int = 15, 
                            frames: int = 120):
        
        # Sanity check here
        #if not (1 <= num_prompts <= 10):
        #    await ctx.send_response(f"<@{ctx.author.id}>, Please specify a number between 1 and 10 for the number of prompts.")
        #    return
        
        await ctx.send_response(f"<@{ctx.author.id}>, Your video prompts: ``{prompts}``")
        
        path = await make_animation(deforum_settings)


    def make_animation(deforum_settings):
        global config

        settings_json = json.dumps(deforum_settings)
        allowed_params = ';'.join(deforum_settings.keys()).strip(';')

        request = {'settings_json':settings_json, 'allowed_params':allowed_params}
        print(config['URL'])
        try:
            response = requests.post(config['URL'], params=request)
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

        return result#path string


    # Simplified version of parse_key_frames from Deforum
    # this one converts format like `0:(lol), 20:(kek)` to prompts JSON
    def parse_prompts(string, filename='unknown'):
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

    def find_animation(d):
        for f in os.listdir(d):
            if f.endswith('.mp4'):
                return os.path.join(d, f)
        return ''

    def find_settings(d):
        for f in os.listdir(d):
            if f.endswith('.txt'):
                return os.path.join(d, f)
        return ''

    def wrap_value(val:str):
        val = val.strip()
        if len(val) > 0 and not '(' in val and not ')' in val:
            val = f'0: ({val})'
        return val


    # Command work here
    last_usage = {}
    async def deforum(ctx, prompts: str = "", cadence: int = 6, seed = -1, strength: str = "0: (0.65)", preview_mode: bool = False, speed_x: str = "0: (0)", speed_y: str = "0: (0)", speed_z: str = "0: (1.75)", rotate_x:str = "0: (0)", rotate_y: str = "0: (0)", rotate_z: str = "0: (0)", w:int = 1024, h: int = 1024, fps: int = 15, frames: int = 120):
        await bot.tree.sync()
        print('Received a /deforum command!')
        
        sender = ctx.message.author.id
        if sender in last_usage:
            mins = (time.time() - last_usage[sender]) / 60.0
            if mins < config['anim_waittime']:
                await ctx.reply(f"You will be able to make an animation in {config['anim_waittime'] - mins} minutes.")
                return
        
        print(prompts)

        prompts = wrap_value(prompts)
        strength = wrap_value(strength)
        speed_x = wrap_value(speed_x)
        speed_y = wrap_value(speed_y)
        speed_z = wrap_value(speed_z)
        rotate_x = wrap_value(rotate_x)
        rotate_y = wrap_value(rotate_y)
        rotate_z = wrap_value(rotate_z)

        chan = ctx.message.channel

        deforum_settings = {'diffusion_cadence':cadence, 'W':w, 'H':h, 'fps':fps, 'seed':seed, 'frames':frames, 'strength_schedule':strength, 'motion_preview_mode':preview_mode, 'translation_x':speed_x, 'translation_y':speed_y, 'translation_z':speed_z, 'rotation_3d_x':rotate_x, 'rotation_3d_y':rotate_y, 'rotation_3d_z':rotate_z}

        # sanity checks
        if cadence < 6 or w > 1024 or h > 1024 or fps < 1 or frames < 1:
            await ctx.reply("Cadence must be greater than 5, width and height can't be larger than 1024 and fps not less than 1")
            return
        
        if frames / cadence > config['true_frames_limit']:
            await ctx.reply(f"With Cadence {cadence} the length of the animation is limited by {cadence * config['true_frames_limit']} frames")
            return

        if len(prompts) > 0:            
            await ctx.reply('Making a Deforum animation...')
            print('Making the animation')

            path = await make_animation(deforum_settings)

            if len(path) > 0:
                print('Animation made.')
                
                anim_file = find_animation(os.path.abspath(path))
                await ctx.send(file=discord.File(anim_file))
                settings_file = find_settings(os.path.abspath(path))
                
                result_seed = -2
                try:
                    with open(settings_file, 'r', encoding='utf-8') as sttn:
                        result_settings = json.loads(sttn.read())
                    result_seed = result_settings['seed']
                except:
                    ...
                #await ctx.send(file=discord.File(settings_file)) # feature for selected users?
                await ctx.reply(('Your animation is done!' if not preview_mode else 'Your movement preview is done!') + (f' Seed used: {result_seed}' if result_seed != -2 else ''))
                
                # don't waste the credits if preview mode is on
                if not preview_mode:
                    last_usage[sender] = time.time()
            else:
                print('Failed to make an animation!')
                traceback.print_exc()
                await ctx.reply('Sorry, there was an error making the animation!')

def setup(bot):
    bot.add_cog(DeforumCog(bot))
