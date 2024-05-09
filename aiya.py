import asyncio
import discord
import os
import sys
from core import ctxmenuhandler
from core import settings
from core.logging import get_logger
from dotenv import load_dotenv
from core.queuehandler import GlobalQueue


# start up initialization stuff
self = discord.Bot()
intents = discord.Intents.default()
intents.members = True
load_dotenv()
self.logger = get_logger(__name__)

# load extensions
# check files and global variables
settings.startup_check()
settings.files_check()

self.load_extension('core.settingscog')
self.load_extension('core.stablecog')
self.load_extension('core.upscalecog')
self.load_extension('core.identifycog')
self.load_extension('core.infocog')
#self.load_extension('core.metacog')
self.load_extension('core.leaderboardcog')
self.load_extension('core.deforumcog')

use_generate = os.getenv("USE_GENERATE", 'True')
enable_generate = use_generate.lower() in ('true', '1', 't')
if enable_generate:
    print(f"/generate command is ENABLED due to USE_GENERATE={use_generate}")
    self.load_extension('core.generatecog')
else:
    print(f"/generate command is DISABLED due to USE_GENERATE={use_generate}")

self.load_extension('core.chatbotcog')

# stats slash command
@self.slash_command(name='stats', description='How many images have I generated?')
async def stats(ctx):
    print(f"/Stats request -- {ctx.author.name}#{ctx.author.discriminator}")
    with open('resources/stats.txt', 'r') as f:
        data = list(map(int, f.readlines()))
    embed = discord.Embed(title='Art generated', description=f'I have created {data[0]} pictures!', color=discord.Color.random())
    await ctx.respond(embed=embed, delete_after=45.0)


# queue slash command
@self.slash_command(name='queue', description='Check the size of each queue')
async def queue(ctx):
    print(f"/Queue request -- {ctx.author.name}#{ctx.author.discriminator}")
    queue_sizes = GlobalQueue.get_queue_sizes()
    description = '\n'.join([f'{name}: {size}' for name, size in queue_sizes.items()])
    embed = discord.Embed(title='Queue Sizes', description=description, 
                          color=settings.global_var.embed_color)
    await ctx.respond(embed=embed, delete_after=45.0)


# ping slash command
@self.slash_command(name='ping', description='Pong!')
async def ping(ctx):
    print(f"/Ping request ({round(self.latency * 1000)}ms)-- {ctx.author.name}#{ctx.author.discriminator}")
    # check for an existing progression message, if yes delete the previous one
    async for old_msg in ctx.channel.history(limit=15):
        if old_msg.embeds:
            if old_msg.embeds[0].title.startswith("**Pong!** -"):
                await old_msg.delete()
    latency_ms = round(self.latency * 1000)
    title = f'**Pong!** - `{latency_ms}ms`'
    embed = discord.Embed(title=title, color=discord.Color.random())
    await ctx.respond(content=f'<@{ctx.author.id}>', embed=embed, delete_after=10)


# context menu commands
@self.message_command(name="Get Image Info")
async def get_image_info(ctx, message: discord.Message):
    await ctxmenuhandler.get_image_info(ctx, message)


@self.message_command(name=f"Quick Upscale")
async def quick_upscale(ctx, message: discord.Message):
    await ctxmenuhandler.quick_upscale(self, ctx, message)


@self.message_command(name=f"Download Batch")
async def batch_download(ctx, message: discord.Message):
    await ctxmenuhandler.batch_download(ctx, message)


@self.event
async def on_ready():
    self.logger.info(f'Logged in as {self.user.name} ({self.user.id})')
    await self.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name='drawing tutorials.'))
    for guild in self.guilds:
        print(f"I'm active in {guild.id} a.k.a {guild}!")


# fallback feature to delete generations if aiya has been restarted
@self.event
async def on_raw_reaction_add(ctx):
    if ctx.emoji.name == '❌':
        try:
            end_user = f'{ctx.user_id}'
            message = await self.get_channel(ctx.channel_id).fetch_message(ctx.message_id)
            if end_user in message.content and "Queue" not in message.content:
                await message.delete()
            # this is for deleting outputs from /identify
            if message.embeds:
                if message.embeds[0].footer.text == f'{ctx.member.name}#{ctx.member.discriminator}':
                    await message.delete()
        except(Exception,):
            # so console log isn't spammed with errors
            pass


@self.event
async def on_guild_join(guild):
    print(f'Wow, I joined {guild.name}!')


async def shutdown(bot):
    await bot.close()


try:
    self.run(os.getenv('TOKEN'))
except KeyboardInterrupt:
    self.logger.info('Keyboard interrupt received. Exiting.')
    asyncio.run(shutdown(self))
except SystemExit:
    self.logger.info('System exit received. Exiting.')
    asyncio.run(shutdown(self))
except Exception as e:
    self.logger.error(e)
    asyncio.run(shutdown(self))
finally:
    sys.exit(0)
