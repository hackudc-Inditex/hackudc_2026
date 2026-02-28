"""
Re-descarga SOLO los bundles de test a resolución original (sin redimensionar).
Uso: python src/download_hires.py
"""

import asyncio
import aiohttp
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
HIRES_DIR = BASE_DIR / "images" / "bundles_hires"
MAX_CONCURRENT = 10
TIMEOUT = aiohttp.ClientTimeout(total=60)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


async def download_image(session, url, save_path, sem, retries=5):
    if save_path.exists():
        return True
    for attempt in range(retries):
        try:
            async with sem:
                await asyncio.sleep(0.05)
                async with session.get(url, timeout=TIMEOUT, headers=HEADERS) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        continue
                    data = await resp.read()
            save_path.write_bytes(data)
            return True
        except Exception:
            if attempt == retries - 1:
                return False
            await asyncio.sleep(1 * (attempt + 1))
    return False


async def main():
    HIRES_DIR.mkdir(parents=True, exist_ok=True)

    bundles = {b["bundle_asset_id"]: b["bundle_image_url"] for b in load_csv(BASE_DIR / "bundles_dataset.csv")}

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, force_close=True)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for bid, url in bundles.items():
            path = HIRES_DIR / f"{bid}.jpg"
            tasks.append(download_image(session, url, path, sem))

        print(f"Descargando {len(tasks)} bundles en alta resolución...")
        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*tasks)

    ok = sum(results)
    fail = len(results) - ok
    print(f"OK: {ok}, Fallos: {fail}")


if __name__ == "__main__":
    asyncio.run(main())
