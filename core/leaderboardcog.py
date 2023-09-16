import csv
import os
import discord
from discord.ext import commands

class LeaderboardCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @staticmethod
    def check_and_create_csv():
        if not os.path.exists("leaderboard.csv"):
            with open("leaderboard.csv", "w", newline='') as csvfile:
                fieldnames = ["User_ID", "Username", "Image_Count", "Identify_Count", "Upscale_Count", "Generate_Count"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    @staticmethod
    def update_leaderboard(user_id, username, action):
        leaderboard_data = []
        
        #print(f' -- Updating {action} Leaderboard for {username} to ')

        with open("leaderboard.csv", "r", newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                leaderboard_data.append(row)

        user_exists = False
        for entry in leaderboard_data:
            if entry["User_ID"] == str(user_id):
                entry[action] = str(int(entry[action]) + 1)
                print(f' -- Updating {action} Leaderboard for {username} to {entry[action]}')
                entry["Username"] = username
                user_exists = True
                break

        if not user_exists:
            new_entry = {"User_ID": str(user_id), "Username": username, "Image_Count": "0", "Identify_Count": "0", "Upscale_Count": "0", "Generate_Count": "0"}
            new_entry[action] = "1"
            leaderboard_data.append(new_entry)

        with open("leaderboard.csv", "w", newline='') as csvfile:
            fieldnames = ["User_ID", "Username", "Image_Count", "Identify_Count", "Upscale_Count", "Generate_Count"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in leaderboard_data:
                writer.writerow(entry)

    @commands.slash_command(name='leaderboard', description='Show the Leaderboard', guild_only=True)
    async def show_leaderboard(self, ctx):

        print(f'/Leaderboard request -- {ctx.author.name} --')

        try:
            leaderboard_data = []

            with open("leaderboard.csv", "r", newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Validate data
                    if 'Username' in row and all(k in row and row[k].isdigit() for k in ['Image_Count', 'Identify_Count', 'Upscale_Count', 'Generate_Count']):
                        leaderboard_data.append(row)
                    else:
                        continue

            # Sort the leaderboard_data by Image_Count
            leaderboard_data.sort(key=lambda x: int(x["Image_Count"]), reverse=True)

            # Create the leaderboard embed
            embed = discord.Embed(title="🏆 Leaderboard 🏆", description="Top 10 Users by Images", color=0x00ff00)
            for idx, entry in enumerate(leaderboard_data[:10]):  # Show top 10
                embed.add_field(name=f"{idx+1}. {entry['Username']}", value=f"{entry['Image_Count']} images, {entry['Identify_Count']} identifies, {entry['Upscale_Count']} upscales, {entry['Generate_Count']} prompts", inline=False)

            await ctx.send_response(content=f'<@{ctx.author.id}>', embed=embed)

        except Exception as e:
            await ctx.send_response(f"An error occurred!")
            print(f"An error occurred: {e}")

LeaderboardCog.check_and_create_csv()

def setup(bot):
    bot.add_cog(LeaderboardCog(bot))
