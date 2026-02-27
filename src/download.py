"""
Descarga paralela de todas las imágenes (bundles + productos).
Uso: python src/download.py
"""

import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm.asyncio import tqdm_asyncio

BASE_DIR = Path(__file__).resolve().parent.parent
BUNDLES_DIR = BASE_DIR / "images" / "bundles"
PRODUCTS_DIR = BASE_DIR / "images" / "products"
MAX_CONCURRENT = 10
TIMEOUT = aiohttp.ClientTimeout(total=60)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


async def download_image(session: aiohttp.ClientSession, url: str, save_path: Path, sem: asyncio.Semaphore, retries: int = 5):
    if save_path.exists():
        return True
    for attempt in range(retries):
        try:
            async with sem:
                await asyncio.sleep(0.05)  # pequeño delay entre requests
                async with session.get(url, timeout=TIMEOUT, headers=HEADERS) as resp:
                    if resp.status == 429:  # rate limited
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        continue
                    data = await resp.read()
            img = Image.open(BytesIO(data)).convert("RGB")
            img = img.resize((224, 224), Image.LANCZOS)
            img.save(save_path, "JPEG", quality=90)
            return True
        except Exception:
            if attempt == retries - 1:
                return False
            await asyncio.sleep(1 * (attempt + 1))
    return False


async def download_all():
    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
    PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)

    bundles_df = pd.read_csv(BASE_DIR / "bundles_dataset.csv")
    products_df = pd.read_csv(BASE_DIR / "product_dataset.csv")

    tasks = []
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        for _, row in bundles_df.iterrows():
            path = BUNDLES_DIR / f"{row['bundle_asset_id']}.jpg"
            tasks.append(download_image(session, row["bundle_image_url"], path, sem))

        for _, row in products_df.iterrows():
            path = PRODUCTS_DIR / f"{row['product_asset_id']}.jpg"
            tasks.append(download_image(session, row["product_image_url"], path, sem))

        # Contar cuantas faltan
        pending = sum(1 for t in tasks if not isinstance(t, bool))
        print(f"Total: {len(tasks)} imágenes ({len(bundles_df)} bundles + {len(products_df)} productos)")
        print(f"Descargando con {MAX_CONCURRENT} conexiones simultáneas...")
        results = await tqdm_asyncio.gather(*tasks)

    ok = sum(results)
    fail = len(results) - ok
    print(f"\nCompletado: {ok} OK, {fail} fallos")
    if fail > 0:
        print("Ejecuta de nuevo para reintentar los fallos (se saltan las ya descargadas)")


if __name__ == "__main__":
    asyncio.run(download_all())
