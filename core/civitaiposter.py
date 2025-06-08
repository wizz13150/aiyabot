import os
import asyncio
import tempfile
import requests

from playwright.async_api import async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

SESSION_DIR = "civitai_session"
UPLOAD_URL = "https://civitai.com/posts/create"

def get_chrome_profiles(user_data_dir):
    # Helper: list all Chrome profile folders for troubleshooting
    try:
        profiles = []
        for d in os.listdir(user_data_dir):
            if os.path.isdir(os.path.join(user_data_dir, d)) and (d.startswith('Default') or d.startswith('Profile')):
                profiles.append(d)
        return profiles
    except Exception as e:
        print(f"Error listing Chrome profiles: {e}")
        return []

def forget_civitai_session():
    import shutil
    user_data_dir = os.path.join(os.environ["LOCALAPPDATA"], "Google", "Chrome", "User Data")
    chrome_profile = "Default"
    chrome_user_data = os.path.join(user_data_dir, chrome_profile)
    try:
        shutil.rmtree(chrome_user_data)
        print(f"Deleted Chrome profile at {chrome_user_data}")
        return True
    except Exception as e:
        print(f"Failed to delete Chrome profile: {e}")
        return False

async def post_image_to_civitai(image_url, user_id):
    print(f"[START] Civitai posting for user {user_id}")
    print(f"Image URL: {image_url}")

    screenshots = []

    # 1. Download image to temp file
    print("Downloading image...")
    try:
        resp = requests.get(image_url)
        if not resp.ok:
            print("Failed to download image: HTTP " + str(resp.status_code))
            return {"success": False, "error": "Image download failed"}

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(resp.content)
        tmp.close()
        local_img = tmp.name
        print(f"Downloaded image to {local_img}")
    except Exception as e:
        print(f"Exception downloading image: {e}")
        return {"success": False, "error": "Exception during image download"}

    # 2. Playwright logic
    try:
        async with async_playwright() as p:
            print("Launching browser (using YOUR regular Chrome profile)...")
            user_data_dir = os.path.join(os.environ["LOCALAPPDATA"], "Google", "Chrome", "User Data")
            profiles = get_chrome_profiles(user_data_dir)
            print(f"[PROFILE CHECK] Detected Chrome profiles: {profiles}")

            chrome_profile = "Default"
            chrome_user_data = os.path.join(user_data_dir, chrome_profile)
            print(f"[PROFILE USE] Using Chrome profile folder: {chrome_user_data}")

            if not os.path.exists(chrome_user_data):
                print(f"[ERROR] Chrome profile folder does NOT exist: {chrome_user_data}")
                return {"success": False, "error": f"Chrome profile folder not found: {chrome_user_data}"}

            browser = await p.chromium.launch_persistent_context(
                chrome_user_data,
                headless=False,
                args=["--start-maximized"]
            )
            page = await browser.new_page()
            print("Navigating to Civitai upload page...")
            await page.goto(UPLOAD_URL, timeout=30000)
            try:
                # Attend explicitement la présence du champ d’upload
                await page.wait_for_selector('input[type="file"]', timeout=30000)
            except PlaywrightTimeoutError:
                print("[ERROR] Timeout waiting for upload page or dropzone to load.")
                await page.screenshot(path=os.path.join(tempfile.gettempdir(), 'civitai_failed_load.png'))
                return {"success": False, "error": "Timeout waiting for Civitai create post page or dropzone."}
            
            await page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            ss1 = os.path.join(tempfile.gettempdir(), f"civitai_1_loaded.png")
            await page.screenshot(path=ss1)
            screenshots.append(ss1)

            # --- LOGIN CHECK ---
            if "login" in page.url or "signin" in page.url or "auth" in page.url:
                print("[LOGIN REQUIRED] Please log in via the opened browser window.")
                print("\n=== PLEASE LOGIN TO CIVITAI IN THE BROWSER ===\n")
                while True:
                    await asyncio.sleep(3)
                    ss2 = os.path.join(tempfile.gettempdir(), f"civitai_2_login.png")
                    await page.screenshot(path=ss2)
                    screenshots.append(ss2)
                    if UPLOAD_URL in page.url:
                        print("Login detected. Ready to upload.")
                        break
            else:
                print("Already logged in (persistent session detected).")

            ss3 = os.path.join(tempfile.gettempdir(), f"civitai_3_uploadform.png")
            await page.screenshot(path=ss3)
            screenshots.append(ss3)

            # --- UPLOAD IMAGE ---
            print("Trying to trigger dropzone and upload image...")
            try:
                # On essaye de cliquer sur le dropzone pour forcer la création de l'input
                dropzone = await page.query_selector('div:has-text("Drag images here")')
                if dropzone:
                    await dropzone.click()
                    print("Clicked the dropzone to trigger file input.")
                    await asyncio.sleep(1)
                else:
                    print("Dropzone not found by text.")

                # On cherche tous les inputs file présents (même invisibles)
                file_inputs = await page.query_selector_all('input[type="file"]')
                found = False
                for idx, inp in enumerate(file_inputs):
                    try:
                        await inp.set_input_files(local_img)
                        print(f"Image file set in file input #{idx}.")
                        found = True
                        break
                    except Exception as e:
                        print(f"Failed to set file for input #{idx}: {e}")

                if not found:
                    # Si ça ne marche toujours pas, on tente d'injecter l'input via JS (HACK)
                    print("[ERROR] No suitable file input found, injecting one via JS as workaround.")
                    await page.evaluate(
                        """() => {
                            let inp = document.createElement('input');
                            inp.type = 'file';
                            inp.accept = 'image/png,image/jpeg,image/webp';
                            inp.style.display = 'none';
                            document.body.appendChild(inp);
                        }"""
                    )
                    new_input = await page.query_selector('input[type="file"][accept]')
                    if new_input:
                        await new_input.set_input_files(local_img)
                        print("Image uploaded via JS-injected input.")
                    else:
                        print("[ERROR] Even JS-injected input failed.")
                        html_path = os.path.join(tempfile.gettempdir(), 'civitai_failed_load.html')
                        print("[DEBUG] HTML dump path:", html_path)
                        print(f"[DEBUG] HTML dump path: {html_path}")
                        html = await page.content()
                        with open(html_path, "w", encoding="utf-8") as f:
                            f.write(html)
                        await page.screenshot(path=os.path.join(tempfile.gettempdir(), 'civitai_failed_load.png'))
                        return {"success": False, "error": "Could not upload image, even after JS injection."}
            except Exception as e:
                print(f"Failed to set file for image input: {e}")
                return {"success": False, "error": f"Exception during image upload: {e}"}

            await asyncio.sleep(2)
            ss4 = os.path.join(tempfile.gettempdir(), f"civitai_4_image_uploaded.png")
            await page.screenshot(path=ss4)
            screenshots.append(ss4)

            # --- POST AUTOMATICALLY ---
            try:
                print("Searching for publish/post button...")
                await asyncio.sleep(4)
                buttons = await page.query_selector_all('button')
                publish_btn = None
                for btn in buttons:
                    try:
                        text = await btn.inner_text()
                        if text.strip().lower() == "publish":
                            if await btn.is_visible() and await btn.is_enabled():
                                publish_btn = btn
                                break
                    except Exception: continue

                if publish_btn:
                    print("Clicking publish button...")
                    await publish_btn.click()
                    await asyncio.sleep(2)
                else:
                    print("No publish button found, user must submit manually.")
                    return {"success": False, "error": "No publish button detected."}
            except Exception as e:
                print(f"Error clicking post button: {e}")

            ss6 = os.path.join(tempfile.gettempdir(), f"civitai_6_posted.png")
            await page.screenshot(path=ss6)
            screenshots.append(ss6)

            # --- WAIT & RELOAD, EXTRACT POST URL (/posts/xxxxxxx) & IMAGE URL (/images/xxxxxxx) ---
            post_url = None
            img_url = None

            try:
                for i in range(7):  # max 7 essais * 2s ~ 14s
                    await asyncio.sleep(2)
                    await page.reload()
                    url = page.url
                    print(f"Reloaded URL: {url}")

                    # Cherche le bon format d'URL de post
                    if "/posts/" in url and "/edit" not in url:
                        post_url = url
                        # Cherche un lien d'image "civitai.com/images/..."
                        links = await page.query_selector_all('a')
                        for a in links:
                            try:
                                href = await a.get_attribute('href')
                                if href and '/images/' in href:
                                    img_url = "https://civitai.com" + href
                                    break
                            except Exception: continue
                        break
                print(f"Final POST url: {post_url}")
                print(f"Image page URL: {img_url}")
            except Exception as e:
                print(f"Error extracting post/image URLs: {e}")

            await asyncio.sleep(2)
            return {
                "success": True,
                "screenshots": screenshots,
                "post_url": post_url,
                "main_img_url": img_url
            }

    except Exception as e:
        print(f"Playwright error: {e}")
        return {"success": False, "error": f"Playwright error: {e}"}
    finally:
        try:
            os.unlink(local_img)
            print(f"Temp image file {local_img} deleted.")
        except Exception as e:
            print(f"Could not delete temp image: {e}")
