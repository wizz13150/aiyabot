import discord
import random
import re
import os
from discord.ui import InputText, Modal, View

from core import ctxmenuhandler
from core import infocog
from core import queuehandler
from core import settings
from core import stablecog
from core import upscalecog
from core import civitaiposter


'''
The input_tuple index reference
input_tuple[0] = ctx
[1] = simple_prompt
[2] = prompt
[3] = negative_prompt
[4] = data_model
[5] = steps
[6] = width
[7] = height
[8] = guidance_scale
[9] = sampler
[10] = seed
[11] = strength
[12] = init_image
[13] = batch
[14] = style
[15] = highres_fix
[16] = clip_skip
[17] = extra_net
[18] = epoch_time
[19] = adetailer
[20] = scheduler
[21] = distilled_cfg_scale
'''
#[20] = poseref
#[21] = ipadapter

tuple_names = ['ctx', 'simple_prompt', 'prompt', 'negative_prompt', 'data_model', 'steps', 'width', 'height',
               'guidance_scale', 'sampler', 'seed', 'strength', 'init_image', 'batch', 'styles',
               'highres_fix', 'clip_skip', 'extra_net', 'epoch_time', 'adetailer', 'scheduler', 'distilled_cfg_scale']# 'poseref', 'ipadapter'

def serialize_input_tuple(input_tuple):
    # Décompose explicitement chaque champ pour éviter toute perte !
    return {
        'author_id': getattr(input_tuple[0].author, "id", None),
        'simple_prompt': input_tuple[1],
        'prompt': input_tuple[2],
        'negative_prompt': input_tuple[3],
        'data_model': input_tuple[4],
        'steps': input_tuple[5],
        'width': input_tuple[6],
        'height': input_tuple[7],
        'guidance_scale': input_tuple[8],
        'sampler': input_tuple[9],
        'seed': input_tuple[10],
        'strength': input_tuple[11],
        'init_image': input_tuple[12],
        'batch': input_tuple[13],
        'styles': input_tuple[14],
        'highres_fix': input_tuple[15],
        'clip_skip': input_tuple[16],
        'extra_net': input_tuple[17],
        'epoch_time': input_tuple[18],
        'adetailer': input_tuple[19],
        #'poseref': input_tuple[20],
        #'ipadapter': input_tuple[21],
        'scheduler': input_tuple[20],
        'distilled_cfg_scale': input_tuple[21]
    }

def deserialize_input_tuple(data):
    # Fake context pour compatibilité avec DrawView
    class FakeAuthor:
        def __init__(self, id):
            self.id = id
    class FakeCtx:
        def __init__(self, author_id):
            self.author = FakeAuthor(author_id)
    ctx = FakeCtx(data['author_id'])
    return (
        ctx,
        data['simple_prompt'],
        data['prompt'],
        data['negative_prompt'],
        data['data_model'],
        data['steps'],
        data['width'],
        data['height'],
        data['guidance_scale'],
        data['sampler'],
        data['seed'],
        data['strength'],
        data['init_image'],
        data['batch'],
        data['styles'],
        data['highres_fix'],
        data['clip_skip'],
        data['extra_net'],
        data['epoch_time'],
        data['adetailer'],
        #data['poseref'],
        #data['ipadapter'],
        data['scheduler'],
        data['distilled_cfg_scale']
    )

class DrawModal(Modal):
    def __init__(self, input_tuple) -> None:
        super().__init__(title="Change Prompt!")
        self.input_tuple = input_tuple
        original_seed = input_tuple[10]


        # run through mod function to get clean negative since I don't want to add it to stablecog tuple
        self.clean_negative = input_tuple[3]
        if settings.global_var.negative_prompt_prefix:
            mod_results = settings.prompt_mod(input_tuple[2], input_tuple[3])
            if settings.global_var.negative_prompt_prefix and mod_results[0] == "Mod":
                self.clean_negative = mod_results[3]

        self.add_item(
            InputText(
                label='Input your new prompt',
                value=input_tuple[1],
                style=discord.InputTextStyle.long
            )
        )
        self.add_item(
            InputText(
                label='Input your new negative prompt (optional)',
                style=discord.InputTextStyle.long,
                value=self.clean_negative,
                required=False
            )
        )
        self.add_item(
            InputText(
                label='Keep seed? Delete to randomize',
                style=discord.InputTextStyle.short,
                value=str(original_seed),
                required=False
            )
        )

        # set up parameters for full edit mode. first get model display name
        display_name = 'Default'
        index_start = 5
        for model in settings.global_var.model_info.items():
            if model[1][0] == input_tuple[4]:
                display_name = model[0]
                break
        # expose each available (supported) option, even if output didn't use them
        ex_params = f'data_model:{display_name}'
        for index, value in enumerate(tuple_names[index_start:], index_start):
            if index == 10 or 12 <= index <= 13 or index == 15:
                continue
            ex_params += f'\n{value}:{input_tuple[index]}'

        self.add_item(
            InputText(
                label='Extended edit (for advanced user!)',
                style=discord.InputTextStyle.long,
                value=ex_params,
                required=False
            )
        )

    async def callback(self, interaction: discord.Interaction):
        # update the tuple with new prompts
        pen = list(self.input_tuple)
        # dedup ensures that if user added lora/hypernet manually to edited prompt
        # it is not duplicated from previous "non-simple" prompt on replacement
        pen[2] = settings.extra_net_dedup(pen[2].replace(pen[1], self.children[0].value))
        pen[1] = self.children[0].value
        pen[3] = self.children[1].value

        # update the tuple new seed (random if invalid value set)
        try:
            pen[10] = int(self.children[2].value)
        except ValueError:
            pen[10] = random.randint(0, 0xFFFFFFFF)
        if (self.children[2].value == "-1") or (self.children[2].value == ""):
            pen[10] = random.randint(0, 0xFFFFFFFF)

        # prepare a validity checker
        new_model, new_token, bad_input = '', '', ''
        model_found = False
        invalid_input = False
        infocog_view = infocog.InfoView()
        net_multi, new_net_multi = 0.85, 0
        embed_err = discord.Embed(title="I can't redraw this!", description="")
        # if extra network is used, find the multiplier
        if pen[17]:
            if pen[17] in pen[2]:
                net_multi = re.search(f'<lora:{re.escape(pen[17])}:(.*?)>', pen[2]).group(1)

        if settings.global_var.size_range:
            max_size = settings.global_var.size_range
        else:
            max_size = settings.global_var.size_range_exceed

        # iterate through extended edit for any changes
        for line in self.children[3].value.split('\n'):
            if 'data_model:' in line:
                new_model = line.split(':', 1)[1]
                # if keeping the "Default" model, don't attempt a model swap
                if new_model == 'Default':
                    pass
                else:
                    for model in settings.global_var.model_info.items():
                        if model[0] == new_model:
                            pen[4] = model[1][0]
                            model_found = True
                            # grab the new activator token
                            new_token = f'{model[1][3]} '.lstrip(' ')
                            break
                    if not model_found:
                        embed_err.add_field(name=f"`{line.split(':', 1)[1]}` is not found.",
                                            value="I used the info command for you! Try one of these models!")
                        await interaction.response.send_message(embed=embed_err, ephemeral=True)
                        await infocog.InfoView.button_model(infocog_view, '', interaction)
                        return

            if 'steps:' in line:
                max_steps = settings.read('% s' % pen[0].channel.id)['max_steps']
                if 0 < int(line.split(':', 1)[1]) <= max_steps:
                    pen[5] = line.split(':', 1)[1]
                else:
                    invalid_input = True
                    embed_err.add_field(name=f"`{line.split(':', 1)[1]}` steps is beyond the boundary!",
                                        value=f"Keep steps between `0` and `{max_steps}`.", inline=False)
            if 'width:' in line:
                try:
                    pen[6] = [x for x in max_size if x == int(line.split(':', 1)[1])][0]
                except(Exception,):
                    invalid_input = True
                    embed_err.add_field(name=f"`{line.split(':', 1)[1]}` width is no good! These widths I can do.",
                                        value=', '.join(['`%s`' % x for x in max_size]),
                                        inline=False)
            if 'height:' in line:
                try:
                    pen[7] = [x for x in max_size if x == int(line.split(':', 1)[1])][0]
                except(Exception,):
                    invalid_input = True
                    embed_err.add_field(name=f"`{line.split(':', 1)[1]}` height is no good! These heights I can do.",
                                        value=', '.join(['`%s`' % x for x in max_size]),
                                        inline=False)
            if 'guidance_scale:' in line:
                try:
                    pen[8] = float(line.split(':', 1)[1].replace(",", "."))
                except(Exception,):
                    invalid_input = True
                    embed_err.add_field(
                        name=f"`{line.split(':', 1)[1]}` is not valid for the guidance scale!",
                        value='Make sure you enter a number. Example: 5.5', inline=False)
            if 'distilled_cfg_scale:' in line:
                try:
                    pen[21] = float(line.split(':', 1)[1].replace(",", "."))
                    print 
                except Exception:
                    invalid_input = True
                    embed_err.add_field(
                        name=f"`{line.split(':', 1)[1]}` is not valid for distilled cfg scale!",
                        value='Make sure you enter a number. Example: 3.5', inline=False)
            if 'sampler:' in line:
                if line.split(':', 1)[1] in settings.global_var.sampler_names:
                    pen[9] = line.split(':', 1)[1]
                else:
                    invalid_input = True
                    embed_err.add_field(name=f"`{line.split(':', 1)[1]}` is unrecognized. I know of these samplers!",
                                        value=', '.join(['`%s`' % x for x in settings.global_var.sampler_names]),
                                        inline=False)
            if 'scheduler:' in line:
                if line.split(':', 1)[1] in settings.global_var.scheduler_names:
                    pen[20] = line.split(':', 1)[1]
                else:
                    invalid_input = True
                    embed_err.add_field(name=f"`{line.split(':', 1)[1]}` is unrecognized. I know of these schedulers!",
                                        value=', '.join(['`%s`' % x for x in settings.global_var.scheduler_names]),
                                        inline=False)
            if 'strength:' in line:
                try:
                    pen[11] = float(line.split(':', 1)[1].replace(",", "."))
                except(Exception,):
                    invalid_input = True
                    embed_err.add_field(name=f"`{line.split(':', 1)[1]}` is not valid for strength!.",
                                        value='Make sure you enter a number (preferably between 0.0 and 1.0).',
                                        inline=False)
            if 'styles:' in line:
                if line.split(':', 1)[1] in settings.global_var.style_names.keys():
                    pen[14] = line.split(':', 1)[1]
                else:
                    embed_err.add_field(name=f"`{line.split(':', 1)[1]}` isn't my style.",
                                        value="I've pulled up the styles list for you from the info command!")
                    await interaction.response.send_message(embed=embed_err, ephemeral=True)
                    await infocog.InfoView.button_style(infocog_view, '', interaction)
                    return
            if 'adetailer:' in line:
                value = line.split(':', 1)[1].strip()
                valid_choices = ['None', 'Faces', 'Hands', 'Faces+Hands', 'Details++']
                if value in valid_choices:
                    pen[19] = value
                else:
                    invalid_input = True
                    embed_err.add_field(name=f"`{value}` is not valid for adetailer!",
                                        value=f'Make sure you enter one of the following: `{", ".join(valid_choices)}`.',
                                        inline=False)
            if 'clip_skip:' in line:
                try:
                    pen[16] = [x for x in range(1, 14, 1) if x == int(line.split(':', 1)[1])][0]
                except(Exception,):
                    invalid_input = True
                    embed_err.add_field(name=f"`{line.split(':', 1)[1]}` is too much CLIP to skip!",
                                        value='The range is from `1` to `12`.', inline=False)
            if 'extra_net:' in line:
                if line.count(':') == 2:
                    net_check = re.search(':(.*):', line).group(1)
                    if net_check in settings.global_var.extra_nets:
                        pen[17] = line.split(':', 1)[1]
                elif line.count(':') == 1 and line.split(':', 1)[1] in settings.global_var.extra_nets:
                    pen[17] = line.split(':', 1)[1]
                else:
                    embed_err.add_field(name=f"`{line.split(':', 1)[1]}` is an unknown extra network!",
                                        value="I used the info command for you! Please review the hypernets and LoRAs.")
                    await interaction.response.send_message(embed=embed_err, ephemeral=True)
                    await infocog.InfoView.button_hyper(infocog_view, '', interaction)
                    return

        # stop and give a useful message if any extended edit values aren't recognized
        if invalid_input:
            await interaction.response.send_message(embed=embed_err, ephemeral=True)
        else:
            # run through mod function if any moderation values are set in config
            new_clean_negative = ''
            if settings.global_var.prompt_ban_list or settings.global_var.prompt_ignore_list or settings.global_var.negative_prompt_prefix:
                mod_results = settings.prompt_mod(self.children[0].value, self.children[1].value)
                if settings.global_var.prompt_ban_list and mod_results[0] == "Stop":
                    await interaction.response.send_message(f"I'm not allowed to draw the word {mod_results[1]}!", ephemeral=True)
                    return
                if settings.global_var.prompt_ignore_list or settings.global_var.negative_prompt_prefix and mod_results[0] == "Mod":
                    if settings.global_var.display_ignored_words == "False":
                        pen[1] = mod_results[1]
                    pen[2] = mod_results[1]
                    pen[3] = mod_results[2]
                    new_clean_negative = mod_results[3]

            # update the prompt again if a valid model change is requested
            if model_found:
                pen[2] = new_token + pen[1]
            # figure out what extra_net was used
            if pen[17] != 'None':
                pen[2], pen[17], new_net_multi = settings.extra_net_check(pen[2], pen[17], net_multi)
            channel = '% s' % pen[0].channel.id
            pen[2] = settings.extra_net_defaults(pen[2], channel)
            # set batch to 1
            #if settings.global_var.batch_buttons == "False":
            #    pen[13] = [1, 1]

            # the updated tuple to send to queue
            prompt_tuple = tuple(pen)
            draw_dream = stablecog.StableCog(self)

            # message additions if anything was changed
            prompt_output = f'\nNew prompt: ``{pen[1]}``'
            if new_clean_negative != '' and new_clean_negative != self.clean_negative:
                prompt_output += f'\nNew negative prompt: ``{new_clean_negative}``'
            if str(pen[4]) != str(self.input_tuple[4]):
                prompt_output += f'\nNew model: ``{new_model}``'
            index_start = 5
            for index, value in enumerate(tuple_names[index_start:], index_start):
                if index == 13 or index == 15 or index == 17:
                    continue
                if str(pen[index]) != str(self.input_tuple[index]):
                    prompt_output += f'\nNew {value}: ``{pen[index]}``'
            if str(pen[17]) != 'None':
                if str(pen[17]) != str(self.input_tuple[17]) and new_net_multi != net_multi or new_net_multi != net_multi:
                    prompt_output += f'\nNew extra network: ``{pen[17]}`` (multiplier: ``{new_net_multi}``)'
                elif str(pen[17]) != str(self.input_tuple[17]):
                    prompt_output += f'\nNew extra network: ``{pen[17]}``'

            print(f'Redraw -- {interaction.user.name}#{interaction.user.discriminator} -- Prompt: {pen[1]}')

            # check queue again, but now we know user is not in queue
            if queuehandler.GlobalQueue.dream_thread.is_alive():
                queuehandler.GlobalQueue.queue.append(queuehandler.DrawObject(stablecog.StableCog(self), *prompt_tuple, DrawView(prompt_tuple)))
            else:
                await queuehandler.process_dream(draw_dream, queuehandler.DrawObject(stablecog.StableCog(self), *prompt_tuple, DrawView(prompt_tuple)))
            await interaction.response.send_message(f'<@{interaction.user.id}>, {settings.messages()}\nQueue: ``{len(queuehandler.GlobalQueue.queue)}``{prompt_output}')

class DrawView(View):
    def __init__(self, input_tuple):
        super().__init__(timeout=None)
        self.input_tuple = input_tuple
        if isinstance(self.input_tuple, tuple): # only check batch if we are actually a real view
            batch = input_tuple[13]
            batch_count = batch[0] * batch[1]
            if batch_count > 1:
                download_menu = DownloadMenu(input_tuple[18], input_tuple[10], batch_count, input_tuple)
                download_menu.callback = download_menu.callback
                self.add_item(download_menu)
                upscale_menu = UpscaleMenu(input_tuple[18], input_tuple[10], batch_count, input_tuple)
                upscale_menu.callback = upscale_menu.callback
                self.add_item(upscale_menu)

    # the 🖋 button will allow a new prompt and keep same parameters for everything else
    @discord.ui.button(
        custom_id="button_re-prompt",
        emoji="🖋",
        label="Edit")
    async def button_draw(self, button, interaction):
        buttons_free = True
        try:
            # check if the output is from the person who requested it
            if settings.global_var.restrict_buttons == 'True':
                if interaction.user.id != self.input_tuple[0].author.id:
                    buttons_free = False
            if buttons_free:
                # if there's room in the queue, open up the modal
                user_queue_limit = settings.queue_check(interaction.user)
                if queuehandler.GlobalQueue.dream_thread.is_alive():
                    if user_queue_limit == "Stop":
                        await interaction.response.send_message(content=f"Please wait! You're past your queue limit of {settings.global_var.queue_limit}.", ephemeral=True)
                    else:
                        await interaction.response.send_modal(DrawModal(self.input_tuple))
                else:
                    await interaction.response.send_modal(DrawModal(self.input_tuple))
            else:
                await interaction.response.send_message("You can't use other people's 🖋!", ephemeral=True)
        except Exception as e:
            print('The pen button broke: ' + str(e))
            # if interaction fails, assume it's because aiya restarted (breaks buttons)
            button.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send("I may have been restarted. This button no longer works.", ephemeral=True)

    # the 🎲 button will take the same parameters for the image, change the seed, and add a task to the queue
    @discord.ui.button(
        custom_id="button_re-roll",
        emoji="🎲",
        label="Re-roll")
    async def button_roll(self, button, interaction):
        buttons_free = True
        try:
            # check if the output is from the person who requested it
            if settings.global_var.restrict_buttons == 'True':
                if interaction.user.id != self.input_tuple[0].author.id:
                    buttons_free = False
            if buttons_free:
                # update the tuple with a new seed
                new_seed = list(self.input_tuple)
                new_seed[10] = random.randint(0, 0xFFFFFFFF)
                # set batch to 1
                if settings.global_var.batch_buttons == "False":
                    new_seed[13] = [1, 1]
                seed_tuple = tuple(new_seed)

                print(f'Reroll -- {interaction.user.name}#{interaction.user.discriminator} -- Prompt: {seed_tuple[1]}')

                # set up the draw dream and do queue code again for lack of a more elegant solution
                draw_dream = stablecog.StableCog(self)
                user_queue_limit = settings.queue_check(interaction.user)
                if queuehandler.GlobalQueue.dream_thread.is_alive():
                    if user_queue_limit == "Stop":
                        await interaction.response.send_message(content=f"Please wait! You're past your queue limit of {settings.global_var.queue_limit}.", ephemeral=True)
                    else:
                        queuehandler.GlobalQueue.queue.append(queuehandler.DrawObject(stablecog.StableCog(self), *seed_tuple, DrawView(seed_tuple)))
                else:
                    await queuehandler.process_dream(draw_dream, queuehandler.DrawObject(stablecog.StableCog(self), *seed_tuple, DrawView(seed_tuple)))

                if user_queue_limit != "Stop":
                    await interaction.response.send_message(
                        f'<@{interaction.user.id}>, {settings.messages()}\nQueue: '
                        f'``{len(queuehandler.GlobalQueue.queue)}`` - ``{seed_tuple[1]}``'
                        f'\nNew Seed:``{seed_tuple[10]}``')
            else:
                await interaction.response.send_message("You can't use other people's 🎲!", ephemeral=True)
        except Exception as e:
            print('The dice roll button broke: ' + str(e))
            # if interaction fails, assume it's because aiya restarted (breaks buttons)
            button.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send("I may have been restarted. This button no longer works.", ephemeral=True)

    # the ⬆️ button will upscale the selected image
    @discord.ui.button(
    custom_id="button_upscale",
    emoji="⬆️",
    label="Upscale")
    async def button_upscale(self, button, interaction):
        buttons_free = True
        try:
            # check if the output is from the person who requested it
            if settings.global_var.restrict_buttons == 'True':
                if interaction.user.id != self.input_tuple[0].author.id:
                    buttons_free = False
            if buttons_free:
                # check if we are dealing with a batch or a single image.
                batch = self.input_tuple[13]
                if batch[0] != 1 or batch[1] != 1:
                    await interaction.response.send_message("Use the drop down menu to upscale batch images!", ephemeral=True) # tell user to use dropdown for upscaling
                else:
                    init_image = self.message.attachments[0]
                    ctx = self.input_tuple[0]
                    channel = '% s' % ctx.channel.id
                    settings.check(channel)
                    upscaler_1 = settings.read(channel)['upscaler_1']
                    upscale_tuple = (ctx, '2.0', init_image, upscaler_1, "None", '0.5', '0.0', '0.0', False) # Create defaults for upscale. If desired we can add options to the per channel upscale settings for this.

                    print(f'Upscaling -- {interaction.user.name}#{interaction.user.discriminator}')

                    # set up the draw dream and do queue code again for lack of a more elegant solution
                    draw_dream = upscalecog.UpscaleCog(self)
                    user_queue_limit = settings.queue_check(interaction.user)
                    if queuehandler.GlobalQueue.dream_thread.is_alive():
                        if user_queue_limit == "Stop":
                            await interaction.response.send_message(content=f"Please wait! You're past your queue limit of {settings.global_var.queue_limit}.", ephemeral=True)
                        else:
                            queuehandler.GlobalQueue.queue.append(queuehandler.UpscaleObject(upscalecog.UpscaleCog(self), *upscale_tuple, DeleteView(upscale_tuple)))
                    else:
                        await queuehandler.process_dream(draw_dream, queuehandler.UpscaleObject(upscalecog.UpscaleCog(self), *upscale_tuple, DeleteView(upscale_tuple)))

                    if user_queue_limit != "Stop":
                        await interaction.response.send_message(
                            f'<@{interaction.user.id}>, {settings.messages()}\nQueue: '
                            f'``{len(queuehandler.GlobalQueue.queue)}`` - Upscaling')
            else:
                await interaction.response.send_message("You can't use other people's ⬆️!", ephemeral=True)
        except Exception as e:
            print('The upscale button broke: ' + str(e))
            # if interaction fails, assume it's because aiya restarted (breaks buttons)
            button.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send("I may have been restarted. This button no longer works.", ephemeral=True)

    '''
    @discord.ui.button(
        custom_id="button_apply_details",
        emoji="✨",
        label="Apply Details++")
    async def button_apply_details(self, button, interaction):
        buttons_free = True
        try:
            # Vérification si l'action est réalisée par l'utilisateur qui a demandé l'image
            #if settings.global_var.restrict_buttons == 'True':
            #    if interaction.user.id != self.input_tuple[0].author.id:
            #        buttons_free = False
            
            # Only Wizz can use this button
            if interaction.user.id != 457981967712124948:
                await interaction.response.send_message(
                    "Only Wizz can use this button.",
                    ephemeral=True
                )
                return
            
            if buttons_free:
                # Mise à jour du tuple pour conserver la même seed et modifier ADetailer et Highres_fix
                new_input = list(self.input_tuple)
                new_input[19] = 'Details++'
                new_input[15] = '4x_foolhardy_Remacri'
                input_tuple = tuple(new_input)

                print(f'Apply Details++ -- {interaction.user.name}#{interaction.user.discriminator} -- Prompt: {input_tuple[1]}')

                # Configuration et ajout de la tâche dans la file d'attente
                draw_dream = stablecog.StableCog(self)
                user_queue_limit = settings.queue_check(interaction.user)
                if queuehandler.GlobalQueue.dream_thread.is_alive():
                    if user_queue_limit == "Stop":
                        await interaction.response.send_message(content=f"Please wait! You're past your queue limit of {settings.global_var.queue_limit}.", ephemeral=True)
                    else:
                        queuehandler.GlobalQueue.queue.append(queuehandler.DrawObject(stablecog.StableCog(self), *input_tuple, DrawView(input_tuple)))
                else:
                    await queuehandler.process_dream(draw_dream, queuehandler.DrawObject(stablecog.StableCog(self), *input_tuple, DrawView(input_tuple)))

                if user_queue_limit != "Stop":
                    await interaction.response.send_message(
                        f'<@{interaction.user.id}>, {settings.messages()}\nQueue: '
                        f'``{len(queuehandler.GlobalQueue.queue)}`` - ``{input_tuple[1]}``'
                        f'\nSame Seed:``{input_tuple[10]}``')
            else:
                await interaction.response.send_message("You can't use this button for others' images!", ephemeral=True)
        except Exception as e:
            print(f'The Apply Details++ button broke: {str(e)}')
            # En cas d'échec de l'interaction, désactiver le bouton
            button.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send("I may have been restarted. This button no longer works.", ephemeral=True)
    '''

    # the 📋 button will let you review the parameters of the generation
    @discord.ui.button(
        custom_id="button_review",
        emoji="📋",
        label="Review")
    async def button_review(self, button, interaction):
        print("[DEBUG] Button review clicked.")

        init_url = None
        try:
            attachment = self.message.attachments[0]
            if self.input_tuple[12]:
                init_url = self.input_tuple[12].url
            embed = await ctxmenuhandler.parse_image_info(init_url, attachment.url, "button")
            await interaction.response.send_message(embed=embed, ephemeral=True)
        except Exception as e:
            print(f"[ERROR] The clipboard button broke: {str(e)}.")

            button.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send("I may have been restarted. This button no longer works.\n"
                                            "You can get the image info from the context menu or **/identify**.",
                                            ephemeral=True)

    @discord.ui.button(
        custom_id="button_post_civitai",
        emoji="🌐",
        label="Post2Civitai")
    async def button_post_civitai(self, button, interaction):
        try:
            WIZZ_ID = 457981967712124948

            # Restriction : Only Wizz can clic !
            if interaction.user.id != WIZZ_ID:
                await interaction.response.send_message("Only Wizz can use this button. (;)", ephemeral=True)
                return
                    # Only allow if single image (batch == [1, 1])
            batch = self.input_tuple[13]
            if batch[0] != 1 or batch[1] != 1:
                await interaction.response.send_message("Posting to Civitai is only available for single images.", ephemeral=True)
                return

            # Check if the output is from the person who requested it
            if settings.global_var.restrict_buttons == 'True':
                if interaction.user.id != self.input_tuple[0].author.id:
                    await interaction.response.send_message("You can't post other people's images!", ephemeral=True)
                    return

            await interaction.response.defer(ephemeral=True)
            attachment = self.message.attachments[0]
            image_url = attachment.url

            # Call your async function to post to civitai with playwright (to be implemented)
            result = await civitaiposter.post_image_to_civitai(
                image_url=image_url,
                user_id=interaction.user.id
            )
            if result.get('success'):
                msg = (
                    f"✅ Successfully posted to Civitai!\n"
                    f"[Open post]({result['post_url']})\n"
                    f"[View image]({result['main_img_url']})\n"
                    f"{result['main_img_url']}"
                )
                await interaction.followup.send(msg)
            else:
                await interaction.followup.send(
                    f"❌ Failed to post to Civitai: {result.get('error', 'Unknown error')}", ephemeral=True
                )
        except Exception as e:
            print('The Civitai post button broke: ' + str(e))
            button.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send(
                f"An error occurred: {str(e)}\nTry again or check logs.",
                ephemeral=True
            )

    # the button to delete generated images
    @discord.ui.button(
        custom_id="button_x",
        emoji="❌",
        label="Delete")
    async def delete(self, button, interaction):
        try:
            await interaction.message.delete()
        except(Exception,):
            button.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send("I may have been restarted. This button no longer works.\n"
                                            "You can react with ❌ to delete the image.", ephemeral=True)

class DeleteView(View):
    def __init__(self, input_tuple):
        super().__init__(timeout=None)
        self.input_tuple = input_tuple

    @discord.ui.button(
        custom_id="button_x_solo",
        emoji="❌",
        label="Delete")
    async def delete(self, button, interaction):
        try:
            # check if the output is from the person who requested it
            user_id, user_name = settings.fuzzy_get_id_name(self.input_tuple[0])
            if interaction.user.id == user_id:
                await interaction.message.delete()
            else:
                await interaction.response.send_message("You can't delete other people's images!", ephemeral=True)
        except(Exception,):
            button.disabled = True
            await interaction.response.edit_message(view=self)
            await interaction.followup.send("I may have been restarted. This button no longer works.\n"
                                            "You can react with ❌ to delete the image.", ephemeral=True)

class DownloadMenu(discord.ui.Select):
    def __init__(self, epoch_time, seed, batch_count, input_tuple):
        self.input_tuple = input_tuple
        batch_count = min(batch_count, 25)
        max_values = min(batch_count, 25)
        filename = [f"{epoch_time}-{seed}-{i}.png" for i in range(1, batch_count+1)]
        input_options = [(f, str(i)) for i, f in enumerate(filename, start=1)]
        options = [discord.SelectOption(label=option[1], value=option[0], description=option[0]) for option in input_options]
        super().__init__(custom_id="download_menu", placeholder='Choose images to download...', min_values=1, max_values=max_values, options=options)

    async def callback(self, interaction: discord.Interaction):
        try: 
            buttons_free = True
            if settings.global_var.restrict_buttons == 'True':
                if interaction.user.id != self.input_tuple[0].author.id:
                    buttons_free = False
            if buttons_free:
                files = []
                for value in self.values:
                    image_path = f'{settings.global_var.dir}/{value}'
                    file = discord.File(image_path, f'{value}')
                    files.append(file)

                if files:
                    await interaction.response.send_message(f'<@{interaction.user.id}>, please wait I am fetching your requested images', view=None)
                    blocks = [files[i:i+10] for i in range(0, len(files), 10)]
                    for block in blocks:
                        await interaction.followup.send(f'<@{interaction.user.id}>, Here are the batch files you requested', files=block, view=DeleteView(self.input_tuple))
            else:
                await interaction.response.send_message("You can't download other people's images!", ephemeral=True)

        except Exception as e:
            print('The download menu broke: ' + str(e))
            self.disabled = True
            await interaction.response.edit_message(view=self.view)
            await interaction.followup.send("I may have been restarted. This button no longer works.\n", ephemeral=True)

class UpscaleMenu(discord.ui.Select):
    def __init__(self, epoch_time, seed, batch_count, input_tuple):
        self.input_tuple = input_tuple
        batch_count = min(batch_count, 25)
        filename = [f"{epoch_time}-{seed}-{i}.png" for i in range(1, batch_count+1)]
        input_options = [(f, str(i)) for i, f in enumerate(filename, start=1)]
        options = [discord.SelectOption(label=option[1], value=option[0], description=option[0]) for option in input_options]
        super().__init__(custom_id="upscale_menu", placeholder='Choose images to upscale...', min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        try: 
            buttons_free = True
            if settings.global_var.restrict_buttons == 'True':
                if interaction.user.id != self.input_tuple[0].author.id:
                    buttons_free = False
            if buttons_free:
                partial_path = f'{settings.global_var.dir}/{self.values[0]}'
                full_path = os.path.join(os.getcwd(), partial_path)
                init_image = 'file://' + full_path
                ctx = self.input_tuple[0]
                channel = '% s' % ctx.channel.id
                settings.check(channel)
                upscaler_1 = settings.read(channel)['upscaler_1']
                upscale_tuple = (ctx, '2.0', init_image, upscaler_1, "None", '0.5', '0.0', '0.0', False) # Create defaults for upscale. If desired we can add options to the per channel upscale settings for this.

                print(f'Upscaling request -- {interaction.user.name}#{interaction.user.discriminator} for {partial_path}')

                # set up the draw dream and do queue code again for lack of a more elegant solution
                draw_dream = upscalecog.UpscaleCog(self)
                user_queue_limit = settings.queue_check(interaction.user)
                if queuehandler.GlobalQueue.dream_thread.is_alive():
                    if user_queue_limit == "Stop":
                        await interaction.response.send_message(content=f"Please wait! You're past your queue limit of {settings.global_var.queue_limit}.", ephemeral=True)
                    else:
                        queuehandler.GlobalQueue.queue.append(queuehandler.UpscaleObject(upscalecog.UpscaleCog(self), *upscale_tuple, DeleteView(upscale_tuple)))
                else:
                    await queuehandler.process_dream(draw_dream, queuehandler.UpscaleObject(upscalecog.UpscaleCog(self), *upscale_tuple, DeleteView(upscale_tuple)))

                if user_queue_limit != "Stop":
                    await interaction.response.send_message(
                        f'<@{interaction.user.id}>, {settings.messages()}\nQueue: '
                        f'``{len(queuehandler.GlobalQueue.queue)}`` - Upscaling')
            else:
                await interaction.response.send_message("You can't upscale other people's images!", ephemeral=True)
        
        except Exception as e:
            print('The upscale menu broke: ' + str(e))
            self.disabled = True
            await interaction.response.edit_message(view=self.view)
            await interaction.followup.send("I may have been restarted. This button no longer works.\n", ephemeral=True)
