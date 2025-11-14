import tarfile
from pathlib import Path
from typing import Iterable, Dict, Union

import asyncio

from .exceptions import MissingAsyncSupport

try:
    import aioboto3
except ImportError as e:  # pragma: no cover
    # We raise a user-friendly error at call time instead of import time.
    aioboto3 = None
    _async_import_error = e
else:
    _async_import_error = None


async def _ensure_async_supported():
    if aioboto3 is None:
        raise MissingAsyncSupport(
            "Async functions require 'aioboto3' to be installed. "
            "Install with: pip install aioboto3"
        )


async def download_shot_async(
    bucket: str,
    shot_id: Union[int, str],
    dest_root: Union[str, Path] = "/tmp/shots",
    archive_ext: str = ".tar.gz",
    overwrite: bool = False,
    skip_if_present: bool = True,
) -> Path:
    """
    Async version of download_shot.

    Downloads a single shot archive from S3 using aioboto3 and extracts it.
    """
    await _ensure_async_supported()

    if isinstance(shot_id, int):
        shot_id_str = f"{shot_id:08d}"
    else:
        shot_id_str = str(shot_id)

    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    shot_dir = dest_root / shot_id_str
    archive_path = dest_root / f"{shot_id_str}{archive_ext}"

    if shot_dir.exists():
        if overwrite:
            for p in shot_dir.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(shot_dir.glob("**/*"), reverse=True):
                if p.is_dir():
                    p.rmdir()
        elif skip_if_present:
            return shot_dir
        else:
            from .exceptions import ShotAlreadyExists

            raise ShotAlreadyExists(
                f"Shot directory {shot_dir} already exists. "
                "Use overwrite=True or skip_if_present=True."
            )

    shot_dir.mkdir(parents=True, exist_ok=True)

    key = f"shots/{shot_id_str}{archive_ext}"

    # Async download
    async with aioboto3.client("s3") as s3:
        await s3.download_file(bucket, key, str(archive_path))

    # Extraction is CPU/disk bound; run in threadpool to avoid blocking the loop
    mode = {
        ".tar.gz": "r:gz",
        ".tgz": "r:gz",
        ".tar.bz2": "r:bz2",
        ".tbz2": "r:bz2",
        ".tar": "r:",
    }.get(archive_ext, "r:*")

    def _extract():
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(path=shot_dir)
        archive_path.unlink(missing_ok=True)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _extract)

    return shot_dir


async def download_many_shots_async(
    bucket: str,
    shot_ids: Iterable[Union[int, str]],
    dest_root: Union[str, Path] = "/tmp/shots",
    archive_ext: str = ".tar.gz",
    concurrency: int = 200,
) -> Dict[Union[int, str], Path]:
    """
    Download many shots concurrently using asyncio + aioboto3.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    shot_ids:
        Iterable of shot IDs.
    dest_root:
        Root directory for shot extraction.
    archive_ext:
        Archive extension.
    concurrency:
        Maximum number of shots downloading concurrently.

    Returns
    -------
    Dict[shot_id, Path]
        Mapping from shot_id to local directory Path.
    """
    await _ensure_async_supported()

    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    results: Dict[Union[int, str], Path] = {}

    async def worker(sid):
        async with sem:
            path = await download_shot_async(
                bucket=bucket,
                shot_id=sid,
                dest_root=dest_root,
                archive_ext=archive_ext,
            )
            results[sid] = path

    tasks = [asyncio.create_task(worker(sid)) for sid in shot_ids]

    # Gather tasks and propagate any errors
    await asyncio.gather(*tasks)

    return results