import time
import numpy as np
import torch

def compute_mean(points: torch.Tensor) -> torch.Tensor:
    return points.mean(dim=0)






from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Initial state ----
INITIAL_POINTS = torch.tensor([
    [100.0, 100.0],
    [200.0, 100.0],
    [150.0, 200.0],
    [250.0, 200.0],
])

# ---- JSON fallback (simple) ----
@app.get("/init")
def init():
    mean = compute_mean(INITIAL_POINTS)

    return {
        "points": INITIAL_POINTS.tolist(),
        "mean": mean.tolist()
    }


# ---- Binary endpoint (future-proof) ----
@app.post("/compute_binary")
async def compute_binary(request: Request):
    start = time.time()

    raw = await request.body()

    # Interpret raw bytes as float32 array
    arr = np.frombuffer(raw, dtype=np.float32)
    points = torch.from_numpy(arr.reshape(-1, 2))

    mean = compute_mean(points)

    # Simulate heavier compute (for testing throttling later)
    # time.sleep(0.0)

    result = mean.numpy().astype(np.float32).tobytes()

    elapsed = time.time() - start

    return Response(
        content=result,
        media_type="application/octet-stream",
        headers={
            "X-Compute-Time": str(elapsed)
        }
    )