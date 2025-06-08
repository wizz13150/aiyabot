import asyncio
import discord
import os
import sys
from discord.ext import commands
from core import ctxmenuhandler
from core import settings
from core.mylogging import get_logger
from dotenv import load_dotenv
from core.queuehandler import GlobalQueue
from core.civitaiposter import forget_civitai_session
#from core.mask_server import MaskEditorServer


# Load environment variables
load_dotenv()

# Setup intents and bot instance
intents = discord.Intents.default()
intents.message_content = True  # Make sure this is enabled for reading message content
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)
bot.logger = get_logger(__name__)

# Startup checks
settings.startup_check()
settings.files_check()

# Load extensions
bot.load_extension('core.settingscog')
bot.load_extension('core.stablecog')
bot.load_extension('core.upscalecog')
bot.load_extension('core.identifycog')
bot.load_extension('core.infocog')
bot.load_extension('core.leaderboardcog')
#bot.load_extension('core.deforumcog')

use_generate = os.getenv("USE_GENERATE", 'True')
enable_generate = use_generate.lower() in ('true', '1', 't')
if enable_generate:
    print(f"/generate command is ENABLED due to USE_GENERATE={use_generate}")
    bot.load_extension('core.generatecog')
else:
    print(f"/generate command is DISABLED due to USE_GENERATE={use_generate}")

bot.load_extension('core.chatbotcog')

@bot.command(name="logoffcivitai")
async def logoffcivitai(ctx):
    """Supprime la session Civitai locale (profil Chrome)."""
    try:
        result = forget_civitai_session()
        if result:
            await ctx.send("✅ Civitai session forgotten. You will need to log in again for the next post.")
        else:
            await ctx.send("❌ Failed to forget Civitai session. Profile not found or error.")
    except Exception as e:
        await ctx.send(f"❌ Error while removing session: {e}")

# Stats slash command
@bot.slash_command(name='stats', description='How many images have I generated?')
async def stats(ctx):
    print(f"/Stats request -- {ctx.author.name}#{ctx.author.discriminator}")
    with open('resources/stats.txt', 'r') as f:
        data = list(map(int, f.readlines()))
    embed = discord.Embed(title='Art generated', description=f'I have created {data[0]} pictures!', color=discord.Color.random())
    await ctx.respond(embed=embed, delete_after=45.0)

# Queue slash command
@bot.slash_command(name='queue', description='Check the size of each queue')
async def queue(ctx):
    print(f"/Queue request -- {ctx.author.name}#{ctx.author.discriminator}")
    queue_sizes = GlobalQueue.get_queue_sizes()
    description = '\n'.join([f'{name}: {size}' for name, size in queue_sizes.items()])
    embed = discord.Embed(title='Queue Sizes', description=description, 
                          color=settings.global_var.embed_color)
    await ctx.respond(embed=embed, delete_after=45.0)

# Ping slash command
@bot.slash_command(name='ping', description='Pong!')
async def ping(ctx):
    print(f"/Ping request ({round(bot.latency * 1000)}ms)-- {ctx.author.name}#{ctx.author.discriminator}")
    # Check for an existing progression message, if yes delete the previous one
    async for old_msg in ctx.channel.history(limit=15):
        if old_msg.embeds:
            if old_msg.embeds[0].title.startswith("**Pong!** -"):
                await old_msg.delete()
    latency_ms = round(bot.latency * 1000)
    title = f'**Pong!** - `{latency_ms}ms`'
    embed = discord.Embed(title=title, color=discord.Color.random())
    await ctx.respond(content=f'<@{ctx.author.id}>', embed=embed, delete_after=10)

# Context menu commands
@bot.message_command(name="Get Image Info")
async def get_image_info(ctx, message: discord.Message):
    await ctxmenuhandler.get_image_info(ctx, message)

@bot.message_command(name=f"Quick Upscale")
async def quick_upscale(ctx, message: discord.Message):
    await ctxmenuhandler.quick_upscale(bot, ctx, message)

@bot.message_command(name=f"Download Batch")
async def batch_download(ctx, message: discord.Message):
    await ctxmenuhandler.batch_download(ctx, message)

# Event: on_ready
@bot.event
async def on_ready():
    bot.logger.info(f'Logged in as {bot.user.name} ({bot.user.id})')
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name='drawing tutorials.'))
    await bot.sync_commands()
    for guild in bot.guilds:
        print(f"I'm active in {guild.id} a.k.a {guild}!")

# Event: on_raw_reaction_add
@bot.event
async def on_raw_reaction_add(ctx):
    if ctx.emoji.name == '❌':
        try:
            end_user = f'{ctx.user_id}'
            message = await bot.get_channel(ctx.channel_id).fetch_message(ctx.message_id)
            if end_user in message.content and "Queue" not in message.content:
                await message.delete()
            # This is for deleting outputs from /identify
            if message.embeds:
                if message.embeds[0].footer.text == f'{ctx.member.name}#{ctx.member.discriminator}':
                    await message.delete()
        except(Exception,):
            # So console log isn't spammed with errors
            pass

# Event: on_guild_join
@bot.event
async def on_guild_join(guild):
    print(f'Wow, I joined {guild.name}!')

# Shutdown function
async def shutdown(bot):
    await bot.close()

# Run the bot
try:
    bot.run(os.getenv('TOKEN'))
except KeyboardInterrupt:
    bot.logger.info('Keyboard interrupt received. Exiting.')
    asyncio.run(shutdown(bot))
except SystemExit:
    bot.logger.info('System exit received. Exiting.')
    asyncio.run(shutdown(bot))
except Exception as e:
    bot.logger.error(e)
    asyncio.run(shutdown(bot))
finally:
    sys.exit(0)
