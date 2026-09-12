import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import asyncio
import time
from atproto import (AsyncFirehoseSubscribeReposClient, parse_subscribe_repos_message, models, CAR)

class BlueskyStreamer:
    def __init__(self, output_queue: asyncio.Queue, interval_seconds: float = 10.0):
        self.queue = output_queue
        self.interval_seconds = interval_seconds
        self.last_processed_time = 0.0

    async def _process_commit(self, commit: models.ComAtprotoSyncSubscribeRepos.Commit) -> None:
        if not commit.blocks:
            return

        car = CAR.from_bytes(commit.blocks)

        for op in commit.ops:
            if op.action != 'create' or not op.cid:
                continue

            if not op.path.startswith('app.bsky.feed.post/'):
                continue

            raw_record = car.blocks.get(op.cid)
            if not raw_record:
                continue

            record = models.get_or_create(raw_record, models.AppBskyFeedPost.Record)
            if not record or not isinstance(record, models.AppBskyFeedPost.Record):
                continue

            langs = getattr(record, 'langs', []) or []
            if 'fr' not in langs:
                continue

            text = getattr(record, 'text', '').strip()
            if not text or len(text) < 10:
                continue

            now = time.time()
            if now - self.last_processed_time < self.interval_seconds:
                continue

            self.last_processed_time = now

            rkey = op.path.split('/')[-1]
            item = {"id": str(op.cid), "rkey": rkey, "author": commit.repo, "text": text, "created_at": getattr(record, 'created_at', None)}
            
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull:
                pass

    async def start(self) -> None:
        while True:
            try:
                client = AsyncFirehoseSubscribeReposClient()

                async def on_message_handler(message):
                    commit = parse_subscribe_repos_message(message)
                    if isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
                        await self._process_commit(commit)

                print("Connexion to Firehose...")
                await client.start(on_message_handler)
            except Exception as e:
                print(f"Disconnected ({e}). Reconnexion in 5s...")
                await asyncio.sleep(5.0)
