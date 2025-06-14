# Customized version of the bot, for FORGE. 
Also need to manually install 3 extensions in the webui: 
- ControlNet (Openpose and IPAdapter, for the related /Draw options)
https://github.com/Mikubill/sd-webui-controlnet
- Deforum (for the /Deforum command)
https://github.com/deforum-art/sd-forge-deforum
- ADetailer (for the related /Draw option)
https://github.com/Bing-su/adetailer<br>

Currently dedicated to [ZavyChromaXL](https://civitai.com/models/119229?modelVersionId=490254) <br>
# AIYA

A Discord bot interface for Stable Diffusion

<img src=https://raw.githubusercontent.com/Kilvoctu/kilvoctu.github.io/master/pics/preview.png  width=50% height=50%>

## Usage

# - To generate an image from text, use the /Draw command and include your prompt as the query.

Command options:

![image](https://github.com/wizz13150/aiyabot/assets/22177081/038cbde5-642b-4e0f-9063-c00c0b38b9e8)

Result:

![image](https://github.com/wizz13150/aiyabot/assets/22177081/a8fa1151-0047-4366-b01a-615cdd801b1c)
![image](https://github.com/wizz13150/aiyabot/assets/22177081/97068bcc-431a-43bd-b2be-912119ffd96d)


# - To generate a prompt from a couple of words, use the /Generate command and include your text as the query.

Command options:

![image](https://github.com/wizz13150/aiyabot/assets/22177081/839640e9-7dcd-4b12-9187-0c036a13f8f4)

Result:

It's in 'text-completion' mode, not 'chat' mode. This will complete the entered prompt, but will not rephrase.<br>
Then you can select some options, a style, or LoRas (you can click several times on buttons, for several choices).

![image](https://github.com/wizz13150/aiyabot/assets/22177081/e59d32e6-fe4f-4d43-9d53-9127d96cb393)



# - To generate a video from a text, use the /Deforum command and include your text, with a correct syntax, and options as the query.

Command options :
(preset option is not functionnal yet)
![image](https://github.com/wizz13150/aiyabot/assets/22177081/577f4b6e-306b-49ca-9794-f125de7bfc8f)

Result:
(preset option is not functionnal yet)
![image](https://github.com/wizz13150/aiyabot/assets/22177081/2169d7f4-8db6-42e1-a3f8-46362cc81bac)
![image](https://github.com/wizz13150/aiyabot/assets/22177081/cd66badc-ad6e-4f64-8f6f-87787b5b6208)
![image](https://github.com/wizz13150/aiyabot/assets/22177081/28830477-839a-4762-9cf6-f68bd340e0cc)

Result:

https://github.com/wizz13150/aiyabot/assets/22177081/39dd5090-3622-4f96-a3cf-84b1778f7119


# - To chat with the llama3 model, tag or reply to the bot to chat.
Start with !generate to let the chatbot to prompt and generate an image. (Draft feature, more or less functionnal). <br>
Use '!stop' to stop the current text generation. Use '!reset' to reset the conversation/chat session.

![image](https://github.com/wizz13150/aiyabot/assets/22177081/8a07cf52-bbfd-464c-8911-b5f3a6e8b6e0)


### Currently supported options for /Draw

- random prompt generation
- negative prompt
- swap model/checkpoint (_[see wiki](https://github.com/Kilvoctu/aiyabot/wiki/Model-swapping)_)
- sampling steps
- width/height
- size ratio (some resolution presets)
- CFG scale
- sampling method
- schedule type
- seed
- Web UI styles
- random style selection
- extra networks (hypernetwork, LoRA)
- high-res fix
- CLIP skip
- img2img
- denoising strength
- batch count
- ADetailer (Faces, Hands, or Both. Include a "Details++" choice, generate a 9MPixels image, using an Ultimate SD Upscale workflow)
- Poseref (ControlNet for SDXL, openpose)
- IPAdapter (ControlNet for SDXL, ipadapter)


#### Bonus features

- /settings command - set per-channel defaults for supported options (_[see Notes](https://github.com/Kilvoctu/aiyabot#notes)!_):
  - also can set maximum steps limit and max batch count limit.
  - refresh (update AIYA's options with any changes from Web UI).
- /deforum command - generate an animation from text (preset option is not functionnal yet).
- /identify command - create a caption for your image or get metadata from an image.
- /generate command - generate prompts from text (text-completion) using a GPT2 model.
- /queue command - shows the size of each queue and the 5 next items in queues.
- /leaderboard command - display a user leaderboard.
- /stats command - shows how many /draw commands have been used.
- /info command - basic usage guide, other info, and download batch images.
- /upscale command - resize your image.
- live preview during generation.
- chatbot using gpt4all to run llama3-8B-instruct (using gpt4all to run a gguf version on small Nvidia or AMD cards, or CPU). (Not fully implemented yet, testing)
- buttons - certain outputs will contain buttons.
  - 🖋 - edit prompt, then generate a new image with same parameters.
  - 🎲 - randomize seed, then generate a new image with same parameters.
  - 📋 - view the generated image's information.
  - ⬆️ - upscale the generated image with defaults. Batch grids require use of the drop downs.
  - ❌ - deletes the generated image.
- dropdown menus - batch images produce two drop down menus for the first 25 images.
  - The first menu prompts the bot to send only the images that you select at single images
  - The second menu prompts the bot to upscale the selected image from the batch.
- context menu options - commands you can try on any message.
  - Get Image Info - view information of an image generated by Stable Diffusion.
  - Quick Upscale - upscale an image without needing to set options.
  - Batch Download - download all images of a batch set without needing to specify batch_id and image_id
- [configuration file](https://github.com/Kilvoctu/aiyabot/wiki/Configuration) - can change some of AIYA's operational aspects. 


## Setup requirements

- Set up [AUTOMATIC1111's Stable Diffusion AI Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
  - AIYA is currently tested on commit `20ae71faa8ef035c31aa3a410b707d792c8203a3` of the Web UI.
- Run the Web UI as local host with API (`COMMANDLINE_ARGS= --api`).
- Clone this repo.
- Create a file in your cloned repo called ".env", formatted like so:
```dotenv
# .env
TOKEN = put your bot token here
```
- Run AIYA by running launch.bat (or launch.sh for Linux)

## Deploy with Docker

AIYA can be deployed using Docker.

The docker image supports additional configuration by adding environment variables or config file updates detailed in the [wiki](https://github.com/Kilvoctu/aiyabot/wiki/Configuration).

### Docker run

```bash
docker run --name aiyabot --network=host --restart=always -e TOKEN=your_token_here -e TZ=America/New_York -v ./aiyabot/outputs:/app/outputs -v ./aiyabot/resources:/app/resources -d ghcr.io/kilvoctu/aiyabot:latest
```

Note the following environment variables work with the docker image:

- `TOKEN` - **[Required]** Discord bot token.
- `URL` - URL of the Web UI API. Defaults to `http://localhost:7860`.
- `TZ` - Timezone for the container in the format `America/New_York`. Defaults to `America/New_York`
- `APIUSER` - API username if required for your Web UI instance.
- `APIPASS` - API password if required for your Web UI instance.
- `USER` - Username if required for your Web UI instance.
- `PASS` - Password if required for your Web UI instance.

### Docker compose

- Clone the repo and refer to the `docker-compose.yml` file in the `deploy` directory.
- Rename the `/deploy/.env.example` file to `.env` and update the `TOKEN` variable with your bot token (and any other configuration as desired).
- Run `docker-compose up -d` to start the bot.

## Notes

- [See wiki for notes on additional configuration.](https://github.com/Kilvoctu/aiyabot/wiki/Configuration)
- [See wiki for notes on swapping models.](https://github.com/Kilvoctu/aiyabot/wiki/Model-swapping)
- [📋 requires a Web UI script. Please see wiki for details.](https://github.com/Kilvoctu/aiyabot/wiki/Frequently-asked-questions-and-other-tips-(setup)#the--and-other-png-info-related-commands-are-not-working)
- Ensure AIYA has `bot` and `application.commands` scopes when inviting to your Discord server, and intents are enabled.
- As /settings can be abused, consider reviewing who can access the command. This can be done through Apps -> Integrations in your Server Settings. Read more about /settings [here.](https://github.com/Kilvoctu/aiyabot/wiki/settings-command)
- AIYA uses Web UI's legacy high-res fix method. To ensure this works correctly, in your Web UI settings, enable this option: `For hires fix, use width/height sliders to set final resolution rather than first pass`


## Credits

#### Foundation
AIYA only exists thanks to these awesome people:
- AUTOMATIC1111, and all the contributors to the Web UI repo.
  - https://github.com/AUTOMATIC1111/stable-diffusion-webui
- harubaru, my entryway into Stable Diffusion (with Waifu Diffusion) and foundation for the AIYA Discord bot.
  - https://github.com/harubaru/waifu-diffusion
  - https://github.com/harubaru/discord-stable-diffusion

#### Great Contributors
These people played a large role in AIYA's development in some way:
- solareon, for developing a more sensible way to display and interact with batches of images.
- danstis, for dockerizing AIYA.
- ashen-sensored, for developing a workaround for Discord removing PNG info to image uploads. *edit* Discord is no longer doing this at the moment, but credit is still due.
  - https://github.com/ashen-sensored/sd_webui_stealth_pnginfo 
- gingivere0, for PayloadFormatter class for the original API. Without that, I'd have given up from the start. Also has a great Discord bot as a no-slash-command alternative.
  - https://github.com/gingivere0/dalebot
- You, for using AIYA and contributing with PRs, bug reports, feedback, and more!
