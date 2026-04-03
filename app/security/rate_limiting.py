import time
from fastapi import Request, HTTPException

# config
RATE_LIMIT = 10
WINDOW = 1     

requests_log = {}


def rate_limiter(request: Request):
    ip = request.client.host
    now = time.time()

    if ip not in requests_log:
        requests_log[ip] = []

    # usuń stare requesty
    requests_log[ip] = [
        timestamp for timestamp in requests_log[ip]
        if now - timestamp < WINDOW
    ]

    # sprawdź limit
    if len(requests_log[ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later."
        )

    # dodaj nowy request
    requests_log[ip].append(now)
