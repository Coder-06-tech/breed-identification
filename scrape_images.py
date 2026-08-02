# scrape_images.py
import os
import re
import sys
import time
import argparse
import urllib.request
import urllib.parse
from PIL import Image
import io

def scrape_breed_images(breed_name, limit=50, output_dir=None):
    if not output_dir:
        # Normalize folder name to avoid issues with spaces or special characters
        folder_name = "".join(x for x in breed_name.title() if x.isalnum())
        output_dir = os.path.join("dataset", folder_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"SEARCH: Starting image search for query: '{breed_name}'...")
    print(f"OUTPUT: Output directory: '{output_dir}'")

    # Construct search URL (Bing Images is easy to parse with regex)
    query = f"{breed_name} cow cattle" if "cow" not in breed_name.lower() and "cattle" not in breed_name.lower() else breed_name
    search_url = "https://www.bing.com/images/search?q=" + urllib.parse.quote_plus(query) + "&FORM=HDRSC2"

    req = urllib.request.Request(
        search_url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }
    )

    try:
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"ERROR: Error fetching search page: {e}")
        return

    # Extract image URLs using regex matching murl fields in JSON
    urls = re.findall(r'murl&quot;:&quot;(http[s]?://.*?)&quot;', html)
    urls += re.findall(r'"murl":"(http[s]?://.*?)"', html)
    # Deduplicate
    urls = list(dict.fromkeys(urls))

    print(f"Found {len(urls)} potential image URLs on Bing.")

    downloaded = 0
    for i, url in enumerate(urls):
        if downloaded >= limit:
            break

        print(f" [{downloaded+1}/{limit}] Downloading: {url} ...")
        try:
            img_req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
                }
            )
            # Timeout of 10s to prevent hanging
            with urllib.request.urlopen(img_req, timeout=10) as img_resp:
                img_data = img_resp.read()

            # Verify image content using PIL
            img = Image.open(io.BytesIO(img_data))
            img = img.convert("RGB")
            
            # Save file
            file_path = os.path.join(output_dir, f"{downloaded}.jpg")
            img.save(file_path, "JPEG")
            print(f"   SAVED to {file_path}")
            downloaded += 1
            
            # Brief pause to be respectful
            time.sleep(0.5)
        except Exception as e:
            print(f"   FAILED to download or parse image: {e}")

    print(f"DONE: Successfully downloaded {downloaded} images of '{breed_name}' to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cattle Breed Image Scraper")
    parser.add_argument("--breed", type=str, required=True, help="Breed search query (e.g. 'Gir cow')")
    parser.add_argument("--limit", type=int, default=50, help="Max images to download (default: 50)")
    parser.add_argument("--output", type=str, default=None, help="Output folder (default: dataset/BreedName)")
    args = parser.parse_args()

    scrape_breed_images(args.breed, args.limit, args.output)
