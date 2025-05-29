import requests
from bs4 import BeautifulSoup
import os

BASE_URL = "https://www.angelone.in"
SUPPORT_URL = BASE_URL + "/support"

def get_support_links():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    response = requests.get(SUPPORT_URL, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch support page: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/support") or href.startswith("/faqs") or "support" in href.lower():
            full_url = BASE_URL + href if href.startswith("/") else href
            links.add(full_url)

    return list(links)

def scrape_and_save_pages():
    os.makedirs("data/html", exist_ok=True)
    links = get_support_links()
    print(f"Found {len(links)} support links")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    for i, url in enumerate(links):
        print(f"Scraping {url}")
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                with open(f"data/html/page_{i}.html", "w", encoding="utf-8") as f:
                    f.write(response.text)
            else:
                print(f"Failed to get page {url} status {response.status_code}")
        except Exception as e:
            print(f"Error scraping {url}: {e}")

if __name__ == "__main__":
    scrape_and_save_pages()
