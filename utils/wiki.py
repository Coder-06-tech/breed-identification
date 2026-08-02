# utils/wiki.py
import re
import json
import urllib.request
import urllib.parse

# Global cache for Wikipedia summaries
WIKI_CACHE = {}

def get_wikipedia_summary(breed_name):
    """Fetch summaries and thumbnails from Wikipedia REST API with in-memory caching"""
    if breed_name in WIKI_CACHE:
        return WIKI_CACHE[breed_name]
        
    # Convert camel case to space-separated words
    spaced_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', breed_name)
    
    # Try querying variations
    queries = [f"{spaced_name} cattle", spaced_name, f"{spaced_name} cow"]
    
    for q in queries:
        wiki_title = urllib.parse.quote(q.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_title}"
        
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) BharatPashudhanAI/1.0'}
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.getcode() == 200:
                    res_data = json.loads(response.read().decode())
                    result_info = {
                        "title": res_data.get("title", spaced_name),
                        "summary": res_data.get("extract", "No description available."),
                        "image": res_data.get("thumbnail", {}).get("source", ""),
                        "url": res_data.get("content_urls", {}).get("desktop", {}).get("page", "")
                    }
                    WIKI_CACHE[breed_name] = result_info
                    return result_info
        except Exception as e:
            continue
            
    # Fallback if no Wikipedia page is found
    result_info = {
        "title": spaced_name,
        "summary": f"The {spaced_name} is a breed of cattle. Further detailed characteristics and origin information are managed under the national livestock database.",
        "image": "",
        "url": f"https://en.wikipedia.org/wiki/Special:Search?search={urllib.parse.quote(spaced_name)}"
    }
    WIKI_CACHE[breed_name] = result_info
    return result_info
