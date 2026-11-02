import asyncio
import time
import requests
import httpx

# A pool of 10 real API endpoints to fetch
link = "https://dummyjson.com/posts"
URLS = [f"{link}/{i}" for i in range(1, 11)]
HEADERS = {"User-Agent": "Mozilla/5.0"}


# ==========================================
# 1. SYNCHRONOUS VERSION (Blocking)
# ==========================================

def fetch_sync(url: str):
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    print(f"[Sync]  Fetched: {data['title'][:32]}...")
    return data


def run_sync() -> float:
    print("--- Starting Synchronous Requests (requests) ---")
    start_time = time.perf_counter()

    results = []
    for url in URLS:
        data = fetch_sync(url)
        results.append(data)

    elapsed = time.perf_counter() - start_time
    print(f"Sync total time: {elapsed:.2f}s\n")
    return elapsed


# ==========================================
# 2. ASYNCHRONOUS VERSION (Non-blocking)
# ==========================================

async def fetch_async(url: str, client: httpx.AsyncClient):
    response = await client.get(url, headers=HEADERS)
    data = response.json()
    print(f"[Async] Fetched: {data['title'][:32]}...")
    return data


async def run_async() -> float:
    print("--- Starting Asynchronous Requests (httpx) ---")
    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(fetch_async(url, client)) for url in URLS]
        results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start_time
    print(f"Async total time: {elapsed:.2f}s\n")
    return elapsed


# ==========================================
# 3. Benchmark Execution & Summary
# ==========================================

def main():
    print(f"Benchmarking {len(URLS)} real HTTP API calls...\n")

    sync_time = run_sync()
    async_time = asyncio.run(run_async())

    speedup = (sync_time / async_time) if async_time > 0 else 0.0

    print("================ SUMMARY ================")
    print(f"Synchronous (requests) : {sync_time:.2f}s")
    print(f"Asynchronous (httpx)   : {async_time:.2f}s")
    print(f"Speedup Factor         : {speedup:.1f}x faster")
    print("=========================================")


if __name__ == "__main__":
    main()
