import sys
from pathlib import Path

# Ajoute la racine du projet au PATH Python
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
        """Décode le CAR file et extrait les posts en français."""
        if not commit.blocks:
            return

        # Décodage du bloc de données de la transaction
        car = CAR.from_bytes(commit.blocks)

        for op in commit.ops:
            # On ne conserve que les créations de posts
            if op.action != 'create' or not op.cid:
                continue

            # Vérification qu'il s'agit bien d'un post Bluesky
            if not op.path.startswith('app.bsky.feed.post/'):
                continue

            # Extraction du record depuis les blocs CAR
            raw_record = car.blocks.get(op.cid)
            if not raw_record:
                continue

            # Conversion en modèle Pydantic Bluesky
            record = models.get_or_create(raw_record, models.AppBskyFeedPost.Record)
            if not record or not isinstance(record, models.AppBskyFeedPost.Record):
                continue

            # Filtre : langue française uniquement
            langs = getattr(record, 'langs', []) or []
            if 'fr' not in langs:
                continue

            text = getattr(record, 'text', '').strip()
            if not text or len(text) < 10:
                continue

            # Throttling : 1 post toutes les N secondes
            now = time.time()
            if now - self.last_processed_time < self.interval_seconds:
                continue

            self.last_processed_time = now

            # Transmission à la queue
            # Envoi à la queue SANS BLOQUER
            rkey = op.path.split('/')[-1]
            item = {
                "id": str(op.cid),
                "rkey": rkey,
                "author": commit.repo,
                "text": text,
                "created_at": getattr(record, 'created_at', None)
            }
            
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

                print("⚡ Connexion au Firehose Bluesky en cours...")
                await client.start(on_message_handler)
            except Exception as e:
                print(f"⚠️ Déconnexion ({e}). Reconnexion dans 5s...")
                await asyncio.sleep(5.0)

"""

# --- Bloc de test ---
if __name__ == "__main__":
    async def main():
        test_queue = asyncio.Queue()
        streamer = BlueskyStreamer(output_queue=test_queue, interval_seconds=3.0)

        async def display_worker():
            while True:
                post = await test_queue.get()
                print(f"\n Nouveau post FR capté [{post['id']}]:")
                print(f"   \"{post['text']}\"")
                test_queue.task_done()

        asyncio.create_task(display_worker())
        await streamer.start()

    asyncio.run(main())

"""
