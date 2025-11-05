# CS 6200 – Project 3 (IPC Web Proxy + Cache)
**Author:** ychi62  
**Semester:** Fall 2025

This document explains what I built for Part 1 and Part 2, why I chose this design, how the code flows, and how I tested it. It’s written so you can copy it as-is and convert to PDF.

---

## 1. Summary

- **Part 1 (Sockets):** The Getfile server becomes a proxy. It builds a full URL from a base and a path, queries the origin using **libcurl**, and streams the response back to the client using the **gfserver** API.
  - HTTP **404/403 → `GF_FILE_NOT_FOUND`**, other non-2xx or missing `Content-Length` → **`GF_ERROR`**, valid 2xx with length → **`GF_OK`**.

- **Part 2 (IPC):** Adds a cache daemon process and splits communication into:
  - **Command channel:** POSIX **message queue** (proxy → cache) sending `{ path, shmName, segmentSize }`.
  - **Data channel:** POSIX **shared memory** blocks that the **proxy owns and pre-creates**; the cache writes file bytes into these blocks using semaphores.
  - Ownership rule from the spec is followed: **proxy owns SHM**, **cache owns MQ**. Startup order is flexible (either may start first).

---

## 2. Architecture

### 2.1 Part 1 – Proxy via libcurl

- Client ──► Proxy(gfserver) ──► HEAD (curl)◄── 2xx + Content-Length
- Proxy ──► gfs_sendheader(GF_OK, len)
- Proxy ──► GET (curl) ──► write-callback ──► gfs_send(ctx, chunk) ──► Client

- Status mapping:
  - 404/403 → `GF_FILE_NOT_FOUND`
  - Other non-2xx or missing `Content-Length` → `GF_ERROR`

### 2.2 Part 2 – Proxy & Cache with MQ + SHM

1) Proxy: take SHM segment from pool; init semaphores
2) Proxy: mq_send { path, shmName, segmentSize } ─────────► Cache
3) Cache: simplecache_get(path); mmap shmName
Cache: write { status, fileSize }; post(filelenSem)
4) Proxy: wait(filelenSem); send header (GF_OK or NOT_FOUND)

Loop while bytes_sent < fileSize:
Cache: wait(cacheSem) → pread(fd, segmentSize) → write { readLen, data[] } → post(proxySem)
Proxy: wait(proxySem) → gfs_send(data, readLen) → post(cacheSem)

5) Proxy: recycle SHM segment back into pool


- **Semaphores per SHM segment:**
  - `filelenSem` — cache signals when `{status, fileSize}` is ready.
  - `cacheSem`   — cache writes the next chunk when it acquires this.
  - `proxySem`   — proxy reads/sends a ready chunk when it acquires this.

---

## 3. Design Choices & Trade-offs

- **HEAD then GET (Part 1):** `gfserver` wants a length before sending the body; a `HEAD` is the simplest way to get `Content-Length`. If missing, I return `GF_ERROR` instead of buffering arbitrarily.
- **Message Queue for the command channel:** Small fixed messages, easy multi-producer/multi-consumer, clean teardown with `mq_unlink`.
- **Shared Memory for the data channel:** Zero-copy handoff with back-pressure via `cacheSem`/`proxySem`. Straightforward to reason about at chunk granularity.
- **Segment Pool:** Pre-create N segments (`-n`) each of size `-z`, and recycle them. Saves repeated `shm_open/ftruncate/mmap` overhead. Requires picking a sensible pool size.
- **Startup Order:** Proxy opens MQ on demand; Cache creates/unlinks MQ at start/stop. Either can start first without crashing.

---

## 4. Implementation Overview

### 4.1 Part 1 Files

- **`handle_with_curl.c`**
  - Builds the full URL from the per-thread `server` arg (+ `path`). If the arg lacks a scheme, it prepends `http://`.
  - Phase 1: `CURLOPT_NOBODY=1` to get headers. Map:
    - 404/403 → `GF_FILE_NOT_FOUND`
    - non-2xx or no `Content-Length` → `GF_ERROR`
  - Phase 2: send `GF_OK` header, then `GET` and stream via a write callback that calls `gfs_send`.
  - `curl_global_init` guarded by a static flag and cleaned via `atexit`.

- **`webproxy.c` (Part 1 mode)**
  - Parses `-p` (port), `-t` (threads), `-s` (origin).
  - `gfserver_init`, then:
    - `GFS_WORKER_FUNC = handle_with_curl`
    - `GFS_WORKER_ARG  = server` (per thread)

### 4.2 Part 2 Files

- **`cache-student.h` (student header shared by proxy/cache)**
  - Command struct sent over MQ:
    ```c
    struct {
      char   path[MAX_REQUEST_LEN];
      char   shmName[10];
      size_t segmentSize;
    };
    ```
  - SHM block:
    ```c
    struct {
      sem_t  proxySem, cacheSem, filelenSem;
      char   shmName[10];
      size_t segmentSize, fileSize;
      int    status;      // 200 or 404
      size_t readLen;
      char   data[];      // up to segmentSize bytes
    };
    ```

- **Proxy side (Part 2)**
  - Pre-creates pool `Shm_0`, `Shm_1`, … and stores them in a `steque` guarded by a mutex + condvar.
  - `handle_with_cache`:
    - Pop a segment, init sems (`filelenSem=0`, `cacheSem=1`, `proxySem=0`).
    - `mq_send` the `{ path, shmName, segmentSize }` request.
    - Wait `filelenSem`, then:
      - If `status=200`: `gfs_sendheader(GF_OK, fileSize)` and run the ping-pong loop.
      - If `status=404`: `gfs_sendheader(GF_FILE_NOT_FOUND, 0)`.
    - Recycle the segment (`readLen=0; fileSize=0; push back to steque`).
  - Signal handler: unlink all SHM segments, destroy queue, stop server.

- **Cache daemon (`simplecached.c`)**
  - Creates **MESSAGEQUEUE**, spawns N workers.
  - Worker:
    - `mq_receive` a request, `simplecache_get(path)`.
    - `sendResponse(fd, shmName, segmentSize)`: `shm_open/mmap`, write `{status, fileSize}`, `post(filelenSem)`, then loop:
      - `sem_wait(cacheSem)` → `pread` → write `{ readLen, data }` → `sem_post(proxySem)`.
  - Signal handler: `mq_unlink(MESSAGEQUEUE)` and `simplecache_destroy()`.

---

## 5. Build & Run

```bash
# Build everything
make

# --- Part 1 ---
# Start proxy (choose a port)
./webproxy -p 16652 -t 8 -s https://raw.githubusercontent.com/gt-cs6200/image_data

# Drive it
./gfclient_download -p 16652 -w workload.txt -c .
./gfclient_measure  -p 16652 -w workload.txt -o metrics.txt

# --- Part 2 ---
# Start cache daemon
./simplecached -c locals.txt -t 6

# Start proxy with SHM pool
./webproxy -n 8 -z 8192 -p 20121 -t 9 -s https://ignored-for-part2

# Drive it against the Part 2 proxy
./gfclient_download -p 20121 -w workload.txt -c .

6. Testing (beyond the autograder)

Part 1

Download known 200s; verify total bytes match Content-Length.

Request a known-bad path; confirm GF_FILE_NOT_FOUND.

Use base without scheme; verify http:// is prepended and fetch works.

Part 2

Start orders: cache → proxy and proxy → cache (no crashes).

Vary pool settings -n and -z (e.g., 4×4KiB, 8×8KiB).

Large files to exercise multiple ping-pong rounds.

Multiple concurrent downloads to observe independent worker progress.

Cache miss path: status=404 → filelenSem posted → proxy returns GF_FILE_NOT_FOUND immediately.

Resilience & cleanup

SIGINT/SIGTERM: proxy unlinks SHM and stops; cache unlinks MQ and exits.

If pread/gfs_send fails mid-transfer: loop breaks, segment recycled, no leaks.

Performance sanity

gfclient_measure confirms chunk timing behaves as expected; smaller segments imply more semaphore turns and slightly lower throughput.

7. Error Handling

Part 1

Non-2xx (except 404/403) → GF_ERROR.

Missing Content-Length → GF_ERROR.

Part 2

Cache miss → status=404, post filelenSem, proxy returns GF_FILE_NOT_FOUND.

All system calls checked (shm_open, ftruncate, mmap, sem_init, mq_*). Failures emit concise errors and abort that path safely.

8. Improvements (actionable)

MQ open backoff in proxy: add a short retry loop (exponential backoff) so startup order is even smoother.

Globally unique SHM names: prefix with proxy PID, e.g., Shm_<pid>_<i>, to avoid collisions in multi-proxy deployments.

Timeouts on waits: use sem_timedwait for both sides to avoid indefinite waits if a peer dies.

Verbosity control: compile-time or CLI flag to keep Gradescope output under the limit.

Part 1 mapping table: keep 404/403 as GF_FILE_NOT_FOUND, lump other 4xx/5xx as GF_ERROR, but log exact origin codes for diagnostics.

9. References

libcurl “easy” API man pages (e.g., CURLOPT_*, curl_easy_perform)

POSIX Message Queues (mq_open, mq_send, mq_receive, mq_unlink)

POSIX Shared Memory (shm_open, ftruncate, mmap, shm_unlink)

POSIX Semaphores (sem_init, sem_wait, sem_post)

Provided gfserver.h and starter code comments

Standard Linux man pages

No external code was copy-pasted; structure follows the project spec.
