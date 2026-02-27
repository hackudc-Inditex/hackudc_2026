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
MAX_CONCURRENT = 50
TIMEOUT = aiohttp.ClientTimeout(total=30)


async def download_image(session: aiohttp.ClientSession, url: str, save_path: Path, sem: asyncio.Semaphore, retries: int = 3):
    if save_path.exists():
        return True
    for attempt in range(retries):
        try:
            async with sem:
                async with session.get(url, timeout=TIMEOUT) as resp:
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
            await asyncio.sleep(0.5)
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

        print(f"Descargando {len(tasks)} imágenes ({len(bundles_df)} bundles + {len(products_df)} productos)...")
        results = await tqdm_asyncio.gather(*tasks)

    ok = sum(results)
    fail = len(results) - ok
    print(f"Completado: {ok} OK, {fail} fallos")
    if fail > 0:
        print("Ejecuta de nuevo para reintentar los fallos (se saltan las ya descargadas)")


if __name__ == "__main__":
    asyncio.run(download_all())
