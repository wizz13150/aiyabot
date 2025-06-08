import os
import base64
import contextlib
import discord
import io
import random
import requests
import time
import traceback
import asyncio
import threading
from urllib.parse import quote
from difflib import SequenceMatcher
from PIL import Image, PngImagePlugin
from discord import option
from discord.ui import InputText, Modal, View
from discord.ext import commands
from typing import Optional

from core import utility
from core import queuehandler
from core import settings
from core import viewhandler


class Minigame:
    def __init__(self, host: discord.User, guild: discord.Guild):
        self.host = host
        self.guild = guild
        self.id = random.randint(0, 0xFFFFFFFF)
        self.running = False
        self.view: MinigameView = None
        self.view_last: MinigameView = None

        self.minigame_id: int = None
        self.reveal_prompt: bool = False
        self.prompt: str = None
        self.prompt_adventure: str = None
        self.model_name: str = None
        self.data_model: str = None
        self.adventure: bool = True
        self.game_iteration: int = 0
        self.batch: int = 2
        self.images_base64: list[str] = []
        self.channel: discord.TextChannel = None

        self.restarting = False
        self.guess_count = 0
        self.image_count = 0

    async def start(self, ctx: discord.ApplicationContext | discord.Interaction):
        if self.running == True:
            return
        self.running = True

        loop = asyncio.get_running_loop()
        self.channel = ctx.channel

        loop.create_task(self.next_image_variation(ctx, self.prompt))

    async def stop(self):
        if self.running == False:
            return
        self.running = False

    async def answer(self, ctx: discord.ApplicationContext | discord.Interaction, guess: str):
        loop = asyncio.get_running_loop()
        user = utility.get_user(ctx)
        content = None
        ephemeral = False

        try:
            if self.running == False:
                content = f'<@{user.id}> This game is over. The answer was ``{self.prompt}``.\n'
                if self.adventure and self.prompt != self.prompt_adventure: content += f'The full prompt was ``{self.prompt_adventure}``.\n'
                content += f'There were ``{self.guess_count}`` guesses and ``{self.image_count}`` images generated.\nPress 🖋️ or 🖼️ to continue the minigame.'
                ephemeral = True
                raise Exception()

            prompt = self.prompt.lower()
            guess = self.sanatize(guess).lower()

            self.guess_count += 1
            similarity = SequenceMatcher(None, prompt, guess).ratio()

            if prompt in guess or similarity > 0.9:
                content = f'<@{user.id}> has guessed the answer! The answer was ``{self.prompt}``.\n'
                if self.adventure and self.prompt != self.prompt_adventure: content += f'The full prompt was ``{self.prompt_adventure}``.\n'
                content += f'It took ``{self.guess_count}`` guesses and ``{self.image_count}`` images to get this answer.\nPress 🖋️ or 🖼️ to continue the minigame.'
                await self.stop()
            elif similarity > 0.8:
                content = f'<@{user.id}> tried ``{guess}``. This is really close.'
            elif similarity > 0.65:
                content = f'<@{user.id}> tried ``{guess}``. This is kind of close.'
            elif similarity > 0.4:
                content = f'<@{user.id}> tried ``{guess}``. Maybe you\'re getting somewhere.'
            else:
                content = f'<@{user.id}> tried ``{guess}``. This was not the correct answer.'

        except Exception as e:
            if content == None:
                content = f'<@{user.id}> Something went wrong.\n{e}'
                print(content + f'\n{traceback.print_exc()}')
                ephemeral = True

        if ephemeral:
            delete_after = 30
        else:
            delete_after = None

        if type(ctx) is discord.ApplicationContext:
            loop.create_task(ctx.send_response(content=content, ephemeral=ephemeral, delete_after=delete_after))
        elif type(ctx) is discord.Interaction:
            loop.create_task(ctx.response.send_message(content=content, ephemeral=ephemeral, delete_after=delete_after))

    async def next_image_variation(self, ctx: discord.ApplicationContext | discord.Interaction, prompt: str = None):
        loop = asyncio.get_running_loop()
        user = utility.get_user(ctx)
        guild = utility.get_guild(ctx)
        content = None
        ephemeral = False
        print(f'Minigame Request -- {user.name}#{user.discriminator} -- {self.guild}')

        try:
            # calculate total cost of queued items and reject if there is too expensive
            queue_cost = round(queuehandler.dream_queue.get_user_queue_cost(user.id), self.batch)
            print(f'Estimated total queued cost -- Queue: {queue_cost}')

            if queue_cost > 0.0:
                print(f'Minigame rejected: Already in queue')
                content = f'<@{user.id}> Please wait! You already have things in queue.'
                ephemeral = True
                raise Exception()

            # reset game if it's not running
            generate_random_prompt = False
            if self.running == False:
                self.running = True
                self.restarting = True
                self.guess_count = 0
                self.image_count = 0
                self.host = user
                self.prompt_adventure = None
                if prompt == None:
                    generate_random_prompt = True # user did not input prompt, generate random prompt

            # generate random prompt if one isn't set
            if not self.prompt:
                generate_random_prompt = True

            # update prompt
            if generate_random_prompt:
                prompt = self.sanatize(prompt)
                self.prompt = await loop.run_in_executor(None, self.get_random_prompt)
                self.reveal_prompt = False
            elif prompt:
                self.prompt = self.sanatize(prompt)
                self.reveal_prompt = True

            # try to make prompt more interesting
            if self.adventure and self.prompt_adventure == None:
                self.prompt_adventure = self.prompt
                try:
                    query = quote(self.prompt.strip())
                    def get_request():
                        try:
                            return requests.get(f'https://lexica.art/api/v1/search?q={query}', timeout=5)
                        except Exception as e:
                            return e

                    response = await loop.run_in_executor(None, get_request)
                    if type(response) is not requests.Response:
                        raise response

                    images = response.json()['images']

                    # use random result
                    # find valid results containing the prompt
                    valid_results: list[str] = []
                    for image in images:
                        if self.prompt in image['prompt']:
                            valid_results.append(image['prompt'])

                    # pick a random result out of the valid results
                    if valid_results:
                        self.prompt_adventure = f'({self.prompt}), ' + self.sanatize(valid_results[random.randrange(0, len(valid_results))])
                except Exception as e:
                    print(f'Dream rejected: Random prompt query failed.\n{e}\n{traceback.print_exc()}')
                    content = f'<@{user.id}> Random prompt query failed.'
                    ephemeral = True
                    raise Exception()

            # start image generation
            content = f'<@{self.host.id}> '
            if self.game_iteration == 0:
                content += 'Your minigame is starting... '
            content += settings.global_var.messages[random.randrange(0, len(settings.global_var.messages))]
            ephemeral = False

            if self.adventure:
                prompt = self.prompt_adventure
            else:
                prompt = self.prompt

            queue_length = await self.get_image_variation(ctx, prompt)
            if queue_length == None:
                self.view = self.view_last
                content = f'<@{user.id}> Sorry, I cannot handle this request right now.'
                ephemeral = True
            else:
                content += f' Queue: ``{queue_length}``'

        except Exception as e:
            if content == None:
                content = f'<@{user.id}> Something went wrong.\n{e}'
                print(content + f'\n{traceback.print_exc()}')
                ephemeral = True

        if ephemeral:
            delete_after = 30
        else:
            delete_after = 120

        if type(ctx) is discord.ApplicationContext:
            loop.create_task(ctx.send_response(content=content, ephemeral=ephemeral, delete_after=delete_after))
        elif type(ctx) is discord.Interaction:
            loop.create_task(ctx.response.send_message(content=content, ephemeral=ephemeral, delete_after=delete_after))

    async def give_up(self, ctx: discord.ApplicationContext | discord.Interaction):
        loop = asyncio.get_running_loop()
        user = utility.get_user(ctx)

        if self.running == False:
            content = f'<@{user.id}> This game is over. The answer was ``{self.prompt}``.\n'
            if self.adventure and self.prompt != self.prompt_adventure: content += f'The full prompt was ``{self.prompt_adventure}``.\n'
            content += f'There were ``{self.guess_count}`` guesses and ``{self.image_count}`` images generated.\nPress 🖋️ or 🖼️ to continue the minigame.'
            ephemeral = True
        else:
            content = f'<@{user.id}> has given up. The answer was ``{self.prompt}``.\n'
            if self.adventure and self.prompt != self.prompt_adventure: content += f'The full prompt was ``{self.prompt_adventure}``.\n'
            content += f'There were ``{self.guess_count}`` guesses and ``{self.image_count}`` images generated.\nPress 🖋️ or 🖼️ to continue the minigame.'
            ephemeral = False

        if type(ctx) is discord.ApplicationContext:
            loop.create_task(ctx.send_response(content=content, ephemeral=ephemeral))
        elif type(ctx) is discord.Interaction:
            loop.create_task(ctx.response.send_message(content=content, ephemeral=ephemeral))

        await self.stop()

    async def get_image_variation(self, ctx: discord.ApplicationContext | discord.Interaction, prompt: str):
        loop = asyncio.get_running_loop()

        # get data model and token from checkpoint
        model_name: str = self.model_name
        data_model: str = self.data_model
        token: str = ''
        width = None
        height = None
        for index, (display_name, full_name) in enumerate(settings.global_var.model_names.items()):
            if display_name == model_name or full_name == model_name:
                model_name = display_name
                data_model = full_name
                token = settings.global_var.model_tokens[display_name]
                if width == None:
                    width = settings.global_var.model_resolutions[display_name]
                if height == None:
                    height = settings.global_var.model_resolutions[display_name]
                break

        if self.adventure == False or self.images_base64:
            guidance_scale = round(4.0 + random.random() * 8.0, 2)
            init_url = 'dummy' # minigame only uses cached images
        else:
            guidance_scale = 1.0 # start with a more random image
            init_url = None

        # randomize sampler
        steps = settings.read(self.guild)['default_steps']
        sampler = settings.global_var.sampler_names[random.randrange(0, len(settings.global_var.sampler_names))]
        if sampler in settings.global_var.slow_samplers: steps = int(steps / 2)

        # insertert negative prompt to reduce chance of AI from getting stuck drawing text
        negative = '[text, word, words, language, written, writing, letter, letters, title, signature, watermark, username, artist name]'

        # generate text output
        words = self.prompt.split(' ')
        message = f'``/minigame'

        if model_name != 'Default':
            message += f' checkpoint:{model_name}'

        message += f' adventure:{self.adventure} batch:{self.batch}``\n'

        if len(words) > 1:
            message += f'``({len(words)} words'
        else:
            message += '``(1 word'

        for index, word in enumerate(words):
            if index == 0:
                message += f', {len(word)} letters'
            else:
                message += f' + {len(word)} letters'

        message += ')``'

        random_message = await loop.run_in_executor(None, self.get_random_word, 'resources/minigame-messages.csv')
        if self.restarting:
            message += f' This is a new prompt. {random_message}'
            self.restarting = False
        else:
            message += f' Guess the prompt. {random_message}'

        # create draw object
        draw_object = utility.DrawObject(
            cog=self,
            ctx=ctx,
            prompt=prompt,
            negative=negative,
            model_name=model_name,
            data_model=data_model,
            steps=steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            sampler=sampler,
            seed=random.randint(0, 0xFFFFFFFF),
            strength=round(0.65 + random.random() * 0.35, 2),
            init_url=init_url,
            batch=self.batch,
            style=None,
            facefix=False,
            tiling=False,
            highres_fix=False,
            highres_fix_prompt=None,
            highres_fix_negative=None,
            clip_skip=1,
            script=None,
            message=message,
            write_to_cache=False
        )

        # lower steps if too expensive
        dream_compute_cost = queuehandler.dream_queue.get_dream_cost(draw_object)
        setting_max_compute = settings.read(self.guild)['max_compute']
        if dream_compute_cost > setting_max_compute:
            steps = min(int(float(steps) * (setting_max_compute / dream_compute_cost)), settings.read(self.guild)['max_steps'])

        print(f'prompt: {prompt} negative:{negative} checkpoint:{model_name} sampler:{sampler} steps:{steps} guidance_scale:{guidance_scale} seed:{draw_object.seed} strength:{draw_object.strength} batch:{self.batch}')

        self.view_last = self.view
        draw_object.view = MinigameView(self, draw_object)
        self.view = draw_object.view

        # construct a payload
        payload_prompt = draw_object.prompt
        if token: payload_prompt = f'[[[{token}]]] ((({payload_prompt})))'

        payload = {
            'prompt': payload_prompt,
            'negative_prompt': draw_object.negative,
            'steps': draw_object.steps,
            'width': draw_object.width,
            'height': draw_object.height,
            'cfg_scale': draw_object.guidance_scale,
            'sampler_index': draw_object.sampler,
            'seed': draw_object.seed,
            'seed_resize_from_h': 0,
            'seed_resize_from_w': 0,
            'denoising_strength': draw_object.strength,
            'tiling': draw_object.tiling,
            'n_iter': draw_object.batch,
            'override_settings': {
                # 'sd_model_checkpoint': draw_object.data_model
                'CLIP_stop_at_last_layers': draw_object.clip_skip
            }
        }

        if self.adventure and init_url and self.images_base64:
            # update payload if image_base64 is available
            img_payload = {
                'init_images': self.images_base64,
                'denoising_strength': draw_object.strength
            }
            payload.update(img_payload)

        draw_object.payload = payload

        #increment number of images generated
        settings.increment_stats(self.batch)

        priority = int(settings.read(self.guild)['priority']) + 1
        return queuehandler.dream_queue.process_dream(draw_object, priority)

        # while game_iteration == self.game_iteration:
        #     await asyncio.sleep(0.1)

    def get_random_prompt(self):
        return self.get_random_word('resources/minigame-words.csv')

    def get_random_word(self, filepath: str):
        file_stats = os.stat(filepath)
        file = open(filepath)
        offset = random.randrange(file_stats.st_size)
        file.seek(offset)
        file.readline()
        random_line = file.readline()

        # extra to handle last/first line edge cases
        if len(random_line) == 0: # we have hit the end
            file.seek(0)
            random_line = file.readline() # so we'll grab the first line instead

        random_line.replace('\n', '')
        return random_line.strip()

    def dream(self, queue_object: utility.DrawObject, web_ui: utility.WebUI, queue_continue: threading.Event):
        user = utility.get_user(queue_object.ctx)

        try:
            if self.running == False:
                # minigame has ended, avoid posting another window
                self.view = self.view_last # allow user to use previous view
                queuehandler.upload_queue.process_upload(utility.UploadObject(queue_object=queue_object,
                    content=f'<@{user.id}> The game is over. The queue for new minigame images have been cancelled.', ephemeral=True, delete_after=30
                ))
                return

            # get webui session
            s = web_ui.get_session()
            if s == None:
                # no session, return the object to the queue handler to try again
                queuehandler.dream_queue.process_dream(queue_object, 0, False)
                return

            # construct a payload for data model
            if queue_object.data_model:
                model_payload = {
                    'sd_model_checkpoint': queue_object.data_model
                }
                s.post(url=f'{web_ui.url}/sdapi/v1/options', json=model_payload, timeout=120)

            # safe for global queue to continue
            def continue_queue():
                time.sleep(0.1)
                queue_continue.set()
            threading.Thread(target=continue_queue, daemon=True).start()

            if self.adventure and queue_object.init_url:
                # workaround for batched init_images payload not working correctly on AUTOMATIC1111
                images: list[str] = queue_object.payload['init_images']
                payloads: list[dict] = []
                threads: list[threading.Thread] = []
                responses: list[requests.Response] = []

                queue_object.payload['init_images'] = []

                for index, image in enumerate(images):
                    new_payload = {}
                    new_payload.update(queue_object.payload)
                    new_payload['init_images'] = [image]
                    new_payload['seed'] = int(new_payload['seed']) + index
                    new_payload['n_iter'] = 1
                    payloads.append(new_payload)
                    responses.append(None)

                def img2img(thread_index, thread_payload):
                    try:
                        responses[thread_index] = s.post(url=f'{web_ui.url}/sdapi/v1/img2img', json=thread_payload, timeout=120)
                    except Exception as e:
                        responses[thread_index] = e

                for index, payload in enumerate(payloads):
                    thread = threading.Thread(target=img2img, args=[index, payload], daemon=True)
                    threads.append(thread)

                for thread in threads:
                    thread.start()

                for thread in threads:
                    thread.join()

                for response in responses:
                    if type(response) is not requests.Response:
                        raise response

                response: requests.Response = None
                response_data = None
                for response_fragment in responses:
                    response_fragment_data = response_fragment.json()
                    if response_data == None:
                        response_data = response_fragment_data
                    else:
                        response_data['images'].append(response_fragment_data['images'][0])
                # end of workaround
            else:
                # do normal batched payload
                response = s.post(url=f'{web_ui.url}/sdapi/v1/txt2img', json=queue_object.payload, timeout=120)
                response_data = response.json()

            if self.running == False:
                # minigame has ended, avoid posting another window
                self.view = self.view_last # allow user to use previous view
                queuehandler.upload_queue.process_upload(utility.UploadObject(queue_object=queue_object,
                    content=f'<@{user.id}> The game is over. The queue for new minigame images have been cancelled.', ephemeral=True, delete_after=30
                ))
                return

            queue_object.payload = None
            self.game_iteration += 1

            def post_dream():
                try:
                    # create safe/sanitized filename
                    keep_chars = (' ', '.', '_')
                    file_name = ''.join(c for c in queue_object.prompt if c.isalnum() or c in keep_chars).rstrip()

                    # save local copy of image and prepare PIL images
                    pil_images: list[Image.Image] = []
                    self.images_base64 = []
                    for i, image_base64 in enumerate(response_data['images']):
                        image = Image.open(io.BytesIO(base64.b64decode(image_base64.split(',',1)[0])))
                        pil_images.append(image)

                        image_base64 = 'data:image/png;base64,' + image_base64
                        self.images_base64.append(image_base64)

                        # save png with metadata
                        if settings.global_var.dir != '--no-output':
                            try:
                                epoch_time = int(time.time())
                                file_path = f'{settings.global_var.dir}/{epoch_time}-{queue_object.seed}-{file_name[0:120]}-{i}.png'

                                metadata = PngImagePlugin.PngInfo()
                                metadata.add_text('parameters', response_data['info'])
                                image.save(file_path, pnginfo=metadata)
                                print(f'Saved image: {file_path}')
                            except Exception as e:
                                print(f'Unable to save image: {file_path}\n{traceback.print_exc()}')
                        else:
                            print(f'Received image: {int(time.time())}-{queue_object.seed}-{file_name[0:120]}-{i}.png')

                    # post to discord
                    with contextlib.ExitStack() as stack:
                        buffer_handles = [stack.enter_context(io.BytesIO()) for _ in pil_images]

                        for (pil_image, buffer) in zip(pil_images, buffer_handles):
                            pil_image.save(buffer, 'PNG')
                            buffer.seek(0)

                        files = [discord.File(fp=buffer, filename=f'{queue_object.seed}-{i}.png') for (i, buffer) in enumerate(buffer_handles)]
                        queuehandler.upload_queue.process_upload(utility.UploadObject(queue_object=queue_object,
                            content=f'<@{user.id}> {queue_object.message}', files=files, view=queue_object.view
                        ))
                        queue_object.view = None
                        self.image_count += queue_object.batch

                except Exception as e:
                    self.view = self.view_last # allow user to use previous view
                    content = f'<@{user.id}> ``{queue_object.message}``\nSomething went wrong.\n{e}'
                    print(content + f'\n{traceback.print_exc()}')
                    queuehandler.upload_queue.process_upload(utility.UploadObject(queue_object=queue_object, content=content, delete_after=30))

            threading.Thread(target=post_dream, daemon=True).start()

        except requests.exceptions.RequestException as e:
            # connection error, return items to queue
            time.sleep(5.0)
            web_ui.reconnect()
            queuehandler.dream_queue.process_dream(queue_object, 0, False)
            return

        except Exception as e:
            self.view = self.view_last # allow user to use previous view
            content = f'<@{user.id}> ``{queue_object.message}``\nSomething went wrong.\n{e}'
            print(content + f'\n{traceback.print_exc()}')
            queuehandler.upload_queue.process_upload(utility.UploadObject(queue_object=queue_object, content=content, delete_after=30))

    # sanatize input strings
    def sanatize(self, input: str):
        if input:
            input = input.replace('`', ' ')
            input = input.replace('\n', ' ')
            input = input.strip()
        return input

class MinigameCog(commands.Cog, description='Guess the prompt from the picture minigame.'):
    ctx_parse = discord.ApplicationContext
    def __init__(self, bot: discord.Bot):
        self.bot: discord.Bot = bot

    @commands.slash_command(name = 'minigame', description = 'Starts a minigame where you guess the prompt from a picture.')
    @option(
        'prompt',
        str,
        description='The starting prompt. If provdided, you will handle the prompts instead of me.',
        required=False,
    )
    @option(
        'checkpoint',
        str,
        description='Select the data model for image generation',
        required=False,
        autocomplete=settings.autocomplete_model,
    )
    @option(
        'adventure',
        bool,
        description='Try to make the prompt look more interesting.',
        required=False,
    )
    @option(
        'batch',
        int,
        description='The number of images to generate. This is \'Batch count\', not \'Batch size\'.',
        required=False,
    )
    async def draw_handler(self, ctx: discord.ApplicationContext, *,
                           prompt: Optional[str] = None,
                           checkpoint: Optional[str] = None,
                           adventure: Optional[bool] = None,
                           batch: Optional[int] = None):
        try:
            model_name: str = checkpoint

            loop = asyncio.get_running_loop()
            host = utility.get_user(ctx)
            guild = utility.get_guild(ctx)

            minigame = Minigame(host, guild)
            minigame.prompt = prompt

            if adventure != None: minigame.adventure = adventure
            if batch != None: minigame.batch = max(1, min(3, batch))
            if not model_name: model_name = settings.read(guild)['data_model']

            data_model: str = ''
            for (display_name, full_name) in settings.global_var.model_names.items():
                if display_name == model_name or full_name == model_name:
                    model_name = display_name
                    data_model = full_name

            minigame.model_name = model_name
            minigame.data_model = data_model

            if prompt:
                minigame.reveal_prompt = True
            else:
                minigame.reveal_prompt = False

            loop.create_task(minigame.start(ctx))

        except Exception as e:
            viewhandler.print_exception(e, ctx.interaction, loop)


class MinigameView(View):
    def __init__(self, minigame: Minigame, input_object: utility.DrawObject):
        super().__init__(timeout=None)
        self.minigame = minigame
        self.input_object = input_object

    # the 🖋️ button will allow a new prompt and keep same parameters for everything else
    @discord.ui.button(
        custom_id='minigame_re-prompt',
        emoji='🖋️')
    async def button_draw(self, button: discord.Button, interaction: discord.Interaction):
        loop = asyncio.get_running_loop()
        try:
            if viewhandler.check_interaction_permission(interaction, loop) == False: return

            # get input object
            if not self.input_object:
                loop.create_task(interaction.response.send_message('I may have been restarted. This interaction no longer works.\nPlease start a new minigame using the /minigame command.', ephemeral=True, delete_after=30))
                return

            # only allow interaction for the host
            if self.minigame.running and self.minigame.host.id != interaction.user.id:
                loop.create_task(interaction.response.send_message(f'Only {self.minigame.host.name} may configure the minigame.', ephemeral=True, delete_after=30))
                return

            # only allow interaction with the latest post
            if self.minigame.view != self:
                loop.create_task(interaction.response.send_message('You may only interact with the latest image from the minigame.', ephemeral=True, delete_after=30))
                return

            loop.create_task(interaction.response.send_modal(MinigameEditModal(self, self.minigame)))

        except Exception as e:
            viewhandler.print_exception(e, interaction, loop)

    # the 🏳️ ends the game and reveals the answer
    @discord.ui.button(
        custom_id='minigame_giveup',
        emoji='🏳️')
    async def button_draw_giveup(self, button: discord.Button, interaction: discord.Interaction):
        loop = asyncio.get_running_loop()
        try:
            if viewhandler.check_interaction_permission(interaction, loop) == False: return

            # get minigame
            if not self.minigame:
                loop.create_task(interaction.response.send_message('I may have been restarted. This interaction no longer works.\nPlease start a new minigame using the /minigame command.', ephemeral=True, delete_after=30))
                return

            # only allow interaction for the host
            if self.minigame.host.id != interaction.user.id:
                loop.create_task(interaction.response.send_message(f'Only {self.minigame.host.name} may give up.', ephemeral=True, delete_after=30))
                return

            # only allow interaction with the latest post
            if self.minigame.view != self:
                loop.create_task(interaction.response.send_message('You may only interact with the latest image from the minigame.', ephemeral=True, delete_after=30))
                return

            # run stablecog dream using draw object
            loop.create_task(self.minigame.give_up(interaction))

        except Exception as e:
            viewhandler.print_exception(e, interaction, loop)

    #the button to delete generated images
    @discord.ui.button(
        custom_id='minigame_x',
        emoji='❌')
    async def delete(self, button: discord.Button, interaction: discord.Interaction):
        await viewhandler.user_delete(interaction)

    # the 🖼️ button will take the same parameters for the image, send the original image to init_image, change the seed, and add a task to the queue
    @discord.ui.button(
        label='More Images',
        custom_id='minigame_image-variation',
        emoji='🖼️',
        row=1)
    async def button_draw_variation(self, button: discord.Button, interaction: discord.Interaction):
        loop = asyncio.get_running_loop()
        try:
            if viewhandler.check_interaction_permission(interaction, loop) == False: return

            # get minigame
            if not self.minigame:
                loop.create_task(interaction.response.send_message('I may have been restarted. This interaction no longer works.\nPlease start a new minigame using the /minigame command.', ephemeral=True, delete_after=30))
                return

            # only allow interaction for the host
            #if self.minigame.running and self.minigame.host.id != interaction.user.id:
            #    loop.create_task(interaction.response.send_message(f'Only {self.minigame.host.name} may create new variations.', ephemeral=True, delete_after=30))
            #    return

            # only allow interaction with the latest post
            if self.minigame.view != self:
                loop.create_task(interaction.response.send_message('You may only interact with the latest image from the minigame.', ephemeral=True, delete_after=30))
                return

            # run stablecog dream using draw object
            loop.create_task(self.minigame.next_image_variation(interaction, None))

        except Exception as e:
            viewhandler.print_exception(e, interaction, loop)

    # guess prompt button
    @discord.ui.button(
        label='Guess Prompt',
        custom_id='minigame_guess-prompt',
        emoji='⌨️',
        row=1)
    async def guess_prompt(self, button: discord.Button, interaction: discord.Interaction):
        loop = asyncio.get_running_loop()
        try:
            if viewhandler.check_interaction_permission(interaction, loop) == False: return

            # get minigame
            if not self.minigame:
                loop.create_task(interaction.response.send_message('I may have been restarted. This interaction no longer works.\nPlease start a new minigame using the /minigame command.', ephemeral=True, delete_after=30))
                return

            # only allow interaction for the host
            if self.minigame.running == False:
                content = f'<@{interaction.user.id}> This game is over. Press 🖋️ or 🖼️ to continue the minigame.'
                loop.create_task(interaction.response.send_message(content, ephemeral=True, delete_after=30))
                return

            loop.create_task(interaction.response.send_modal(MinigameAnswerModal(self, self.minigame)))

        except Exception as e:
            viewhandler.print_exception(e, interaction, loop)


class MinigameEditModal(Modal):
    def __init__(self, view: MinigameView, minigame: Minigame) -> None:
        super().__init__(title='Change Prompt!')
        self.view = view
        self.minigame = minigame

        # only reveal the prompt if it was manually inputted
        if minigame.reveal_prompt:
            prompt = minigame.prompt
        else:
            prompt = ''

        self.add_item(
            InputText(
                label='Prompt. Leave empty for a random prompt.',
                value=prompt,
                style=discord.InputTextStyle.short,
                required=False
            )
        )

    async def callback(self, interaction: discord.Interaction):
        loop = asyncio.get_running_loop()
        try:
            if viewhandler.check_interaction_permission(interaction, loop) == False: return

            # obtain URL for the original image
            if not self.minigame:
                loop.create_task(interaction.response.send_message('I may have been restarted. This button no longer works.\nPlease try using 🖼️ on a message containing the full /dream command.', ephemeral=True, delete_after=30))
                return

            # only allow interaction for the host
            if self.minigame.running and self.minigame.host.id != interaction.user.id:
                loop.create_task(interaction.response.send_message(f'Only {self.minigame.host.name} may configure the minigame.', ephemeral=True, delete_after=30))
                return

            prompt = self.children[0].value
            if prompt:
                self.minigame.reveal_prompt = True
            else:
                self.minigame.reveal_prompt = False
                prompt = None

            # run stablecog dream using draw object
            await self.minigame.stop()
            loop.create_task(self.minigame.next_image_variation(interaction, prompt))

        except Exception as e:
            viewhandler.print_exception(e, interaction, loop)


class MinigameAnswerModal(Modal):
    def __init__(self, view: MinigameView, minigame: Minigame) -> None:
        super().__init__(title='Guess Prompt')
        self.view = view
        self.minigame = minigame

        self.add_item(
            InputText(
                label='What\'s your guess?',
                value='',
                style=discord.InputTextStyle.short,
                required=False
            )
        )

    async def callback(self, interaction: discord.Interaction):
        loop = asyncio.get_running_loop()
        try:
            if viewhandler.check_interaction_permission(interaction, loop) == False: return

            # obtain URL for the original image
            if not self.minigame:
                loop.create_task(interaction.response.send_message('I may have been restarted. This button no longer works.\nPlease try using 🖼️ on a message containing the full /dream command.', ephemeral=True, delete_after=30))
                return

            guess = self.children[0].value

            # run stablecog dream using draw object
            loop.create_task(self.minigame.answer(interaction, guess))

        except Exception as e:
            viewhandler.print_exception(e, interaction, loop)


def setup(bot: discord.Bot):
    bot.add_cog(MinigameCog(bot))