"""
UDP bridge: receives JPEG frames over UDP (see sender.py) and fans them out
to WebSocket subscribers on /api/v1/inference/ws-stream.
"""

import socket
import threading
import logging
import asyncio
from typing import List

logger = logging.getLogger(__name__)


class UDPBridge:
    def __init__(self, host: str = "0.0.0.0", port: int = 5005):
        self.host = host
        self.port = port
        self.sock = None
        self.running = False
        self.thread = None
        self.loop = None
        self.queues: List[asyncio.Queue] = []
        self._lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.5)
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        logger.info(f"UDPBridge: Listening on {self.host}:{self.port}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.sock:
            self.sock.close()

    def _listen_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(65535)
                if data:
                    self._broadcast(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"UDPBridge recv error: {e}")

    def _broadcast(self, data: bytes):
        with self._lock:
            for q in self.queues:
                # asyncio.Queue is not thread-safe, so hand the frame to the
                # event loop that owns the queue instead of calling put directly.
                if self.loop:
                    self.loop.call_soon_threadsafe(self._offer, q, data)

    @staticmethod
    def _offer(q: asyncio.Queue, data: bytes):
        """Put a frame on a subscriber queue, dropping the oldest frame if full."""
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(data)

    async def subscribe(self) -> asyncio.Queue:
        # Capture the loop on first subscription
        if not self.loop:
            self.loop = asyncio.get_running_loop()

        q = asyncio.Queue(maxsize=10)  # buffer a few frames
        with self._lock:
            self.queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        with self._lock:
            if q in self.queues:
                self.queues.remove(q)


udp_bridge = UDPBridge()
