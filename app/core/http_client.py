import httpx
from typing import Optional

class GlobalHTTPClient:
    client: Optional[httpx.AsyncClient] = None

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        if cls.client is None:
            # Fallback in case it's called outside the FastAPI lifecycle
            cls.client = httpx.AsyncClient(timeout=10.0)
        return cls.client

    @classmethod
    async def start(cls):
        """Initialize the client when the app starts."""
        cls.client = httpx.AsyncClient(timeout=10.0)

    @classmethod
    async def stop(cls):
        """Close the client cleanly when the app shuts down."""
        if cls.client is not None:
            await cls.client.aclose()
            cls.client = None
            import httpx
from typing import Optional

class GlobalHTTPClient:
    client: Optional[httpx.AsyncClient] = None

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        if cls.client is None:
            cls.client = httpx.AsyncClient(timeout=10.0)
        return cls.client

    @classmethod
    async def start(cls):
        cls.client = httpx.AsyncClient(timeout=10.0)

    @classmethod
    async def stop(cls):
        if cls.client is not None:
            await cls.client.aclose()
            cls.client = None