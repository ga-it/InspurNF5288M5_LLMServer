# LMDeploy on Volta (Inspur NF5288M5, 8×V100 32GB) — Qwen3‑30B‑A3B @ ~38K Context (TurboMind, FP16)

A practical, battle‑tested setup for running **Qwen3‑30B‑A3B** with **~38k context** on an **Inspur NF5288M5** (8× NVIDIA Tesla **V100 32GB SXM2** with NVLink) using **LMDeploy**’s **TurboMind** backend in **FP16** with **tensor parallelism**.

This repository provides:
- A minimal **Dockerfile** tuned for **sm_70 (Volta)** and CUDA 12.1.
- A **docker‑compose** service that exposes an **OpenAI‑compatible API** on port **23333**.
- A sample **.env** describing the key knobs to hit high throughput and long context on V100s.

> Why this exists: Many modern PyTorch stacks are dropping or weakening **Volta** support and/or assume **BF16** and Ampere‑class kernels (or newer). **LMDeploy + TurboMind** continues to run **FP16** efficiently on **sm_70** and plays nicely with NVLink‑connected V100s.

---

## Results in Brief

- **Model**: `Qwen/Qwen3-30B-A3B`
- **Context**: configured to **~38,912 tokens** (see `SESSION_LEN`)
- **Precision**: **FP16** (no BF16 on Volta)
- **Parallelism**: **Tensor Parallel = 8** (one shard per V100)
- **Backend**: **LMDeploy TurboMind** (`serve api_server`)
- **API**: OpenAI‑compatible at `http://<host>:23333/v1/*`

> Subjectively: **blazing fast** for a 30B class model on V100s with long context. Your exact throughput will depend on prompt/response lengths and batching.

---

## Why LMDeploy/TurboMind on V100 (Volta)?

**Volta limitations** you must plan around:
- **No BF16** — many modern kernels & quant paths assume BF16 or Ampere+.  
- **PyTorch deprecations** — Volta support is increasingly neglected; CUDA/PyTorch combos that work are shrinking.
- **Weight‑only quant kernels** in some stacks (e.g., TRT‑LLM/NIM) are **Ampere+ only**, so Volta is excluded from the “fast path”.  
- **FP32 pressure** — when stacks force FP32, memory doubles vs FP16.

**TurboMind advantages for Volta**:
- Mature **FP16** path on **sm_70**.
- **Tensor parallelism** scales cleanly across 8× V100 with NVLink.
- Long‑context **KV‑cache** handling is stable and configurable.
- Simple **OpenAI‑compatible** serving with a small operational footprint.

---

## Hardware & Host

- **Chassis**: Inspur **NF5288M5**
- **CPU**: 2× Intel Xeon Gold 6148 (20‑core, 2.4 GHz)
- **RAM**: 512 GB DDR4
- **GPU**: 8× NVIDIA Tesla **V100 32 GB SXM2** with **NVLink**
- **Host OS**: Debian **testing** (trixie)
- **GPU Driver / CUDA**: Works with **CUDA 12.1** runtime in-container; ensure host driver supports it.

> These settings assume all eight GPUs are linked via NVLink. If you have a different topology, adjust `TENSOR_PARALLEL_SIZE` accordingly.

---

## What’s in this repo

- `lmdeploy-v100-debian-testing-dockerfile` — base image: `nvidia/cuda:12.1.1-runtime-ubuntu22.04`, exposes **23333**, `ENTRYPOINT ["lmdeploy"]`.
- `lmdeploy-v100-debian-testing-docker-compose.yml` — builds an **OpenAI‑compatible** API service on **port 23333**, with sensible **NCCL** and cache settings for NVLink V100.
- `lmdeploy-v100-debian-testing.env` — sample environment configuration (**do not commit real tokens**).

> **Security note:** Replace any placeholder tokens locally. **Never** commit `HUGGING_FACE_HUB_TOKEN` to a public repo.

---

## Quick Start

1. **Prerequisites on the host**
   - NVIDIA driver compatible with **CUDA 12.1** containers.
   - **Docker** + **nvidia‑container‑toolkit** installed and working:
     ```bash
     sudo apt-get update
     sudo apt-get install -y docker.io
     distribution=$(. /etc/os-release;echo $ID$VERSION_ID)        && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg        && curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list |             sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |             sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
     sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
     sudo nvidia-ctk runtime configure --runtime=docker
     sudo systemctl restart docker
     ```

2. **Clone your repo and prepare env**
   ```bash
   git clone <your-repo-url> lmdeploy-volta
   cd lmdeploy-volta

   # Copy the sample env and edit values
   cp lmdeploy-v100-debian-testing.env .env

   # IMPORTANT: set your token & model path
   # HUGGING_FACE_HUB_TOKEN=<your-token>
   # MODEL_PATH=Qwen/Qwen3-30B-A3B  # or a local path under /models
   ```

3. **Prepare model/cache directories (optional but recommended)**
   ```bash
   sudo mkdir -p /data/huggingface /data/lmdeploy/cache /data/lmdeploy/models
   sudo chown -R $USER:$USER /data/huggingface /data/lmdeploy
   ```
   If you want the model stored locally (faster cold starts), download it under `/data/lmdeploy/models` and set `MODEL_PATH=/models/<your-model>` in `.env`.

4. **Build & run**
   ```bash
   docker compose -f lmdeploy-v100-debian-testing-docker-compose.yml --env-file .env up -d --build
   docker logs -f lmdeploy-server  # watch first load
   ```

5. **Health check**
   ```bash
   curl -fsS http://localhost:23333/v1/models | jq
   ```

6. **Test the OpenAI‑compatible chat endpoint**
   ```bash
   curl http://localhost:23333/v1/chat/completions      -H "Content-Type: application/json"      -d '{
       "model": "Qwen3-30B-A3B",
       "messages": [{"role":"user", "content":"In one sentence, tell me why TurboMind is good for Volta."}],
       "temperature": 0.2,
       "max_tokens": 200
     }'
   ```

---

## Key Configuration Knobs (from `.env`)

- `MODEL_PATH` — `Qwen/Qwen3-30B-A3B` or an absolute path mounted to `/models/...`.
- `TENSOR_PARALLEL_SIZE` — `8` for 8 GPUs. Must divide the number of visible GPUs.
- `SESSION_LEN` — `38912` here for ~38k context. Higher context ⇒ more KV cache memory.
- `CACHE_MAX_ENTRY` — `0.8` is conservative for the 30B; tweak if you see cache eviction.
- `CACHE_BLOCK_SEQ_LEN` — `128` works well for long context.
- `QUANT_POLICY` — `0` (no quant). On V100 FP16 is reliable; int8 may help VRAM but can cost speed/quality. Use with care.
- `GPU_MEMORY_UTILIZATION` — `0.90` is a good starting point.
- `BLOCK_SIZE` — `64` tends to balance throughput & latency.
- `HF_*` flags — keep `HF_HUB_ENABLE_HF_TRANSFER=0` on slow or fragile links.

Environment for NVLink V100s (set in compose):
- `CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"`
- `NCCL_P2P_LEVEL=NVL`, `NCCL_IB_DISABLE=1`, `NCCL_DEBUG=WARN`
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`

> Tip: If you scale down to **4 GPUs**, set `TENSOR_PARALLEL_SIZE=4` and reduce `SESSION_LEN` or batch size to stay within VRAM.

---

## Operational Guidance

- **Long context vs throughput**: Large `SESSION_LEN` increases KV‑cache pressure. If you hit OOM or cache thrashing, lower `SESSION_LEN` first.
- **Batching**: Increase request concurrency only after confirming single‑stream stability. Watch VRAM and throughput together.
- **Pin the model**: Use a local `MODEL_PATH` to avoid cold‑start downloads from Hugging Face in production.
- **Healthcheck**: The compose file includes a `/v1/models` healthcheck and a prolonged `start_period` to tolerate first‑load times on 30B.
- **Logs**: `docker logs -f lmdeploy-server` during first load; subsequent restarts should be faster with warm caches.
- **Upgrades**: Favor minor upgrades of LMDeploy over major PyTorch/CUDA jumps on Volta systems.

---
# Build and Compose Commands (using `/opt/lmdeploy`)

## Build the image

```bash
docker build -t lmdeploy-volta \
  -f /opt/lmdeploy/lmdeploy-v100-debian-testing-dockerfile \
  /opt/lmdeploy
```

* **What it does:** Builds from the Dockerfile at `/opt/lmdeploy/...dockerfile` with build context `/opt/lmdeploy`, tagging the image `lmdeploy-volta`.
* **Use when:** First setup or after Dockerfile/base image changes.

## Start the service (compose up)

```bash
docker compose \
  -f /opt/lmdeploy/lmdeploy-v100-debian-testing-docker-compose.yml \
  --env-file /opt/lmdeploy/lmdeploy-v100-debian-testing.env \
  up -d --build
```

* **What it does:** Builds if needed and starts the stack in the background.
* **Why `--env-file`:** Injects model/cache/tuning from your `/opt/lmdeploy/...env`.
* **Why `--build`:** Ensures image is rebuilt if anything changed.

## Stop and remove (compose down)

```bash
docker compose \
  -f /opt/lmdeploy/lmdeploy-v100-debian-testing-docker-compose.yml \
  down
```

* **What it does:** Stops and removes containers and the compose network.
* **Use when:** You want a clean stop without deleting host bind-mounted data.

### Optional cleanups

```bash
# Also remove named volumes created by this stack
docker compose -f /opt/lmdeploy/lmdeploy-v100-debian-testing-docker-compose.yml down -v

# Clean up any orphaned containers if service names changed
docker compose -f /opt/lmdeploy/lmdeploy-v100-debian-testing-docker-compose.yml down --remove-orphans
```

* **Caution on `-v`:** Deletes **named volumes**; bind-mounted host paths (e.g., under `/data`) are not touched.

---

## Troubleshooting

- **`no kernel image is available for execution on the device`**  
  Your image or build targets the wrong compute capability. Ensure `TORCH_CUDA_ARCH_LIST="7.0"` (Volta) and a CUDA runtime that supports your driver.

- **TRT‑LLM/NIM weight‑only quant paths fail or are slow**  
  Many fast kernels are **Ampere+** only. On V100, prefer **TurboMind FP16**.

- **OOM at load or first requests**  
  Reduce `SESSION_LEN`, ensure `TENSOR_PARALLEL_SIZE` matches GPU count, and verify no other GPU jobs are running. Consider setting `GPU_MEMORY_UTILIZATION` down slightly.

- **Inter‑GPU throughput is poor**  
  Check NVLink topology/cabling and confirm NCCL settings. Keep `NCCL_P2P_LEVEL=NVL` and `NCCL_IB_DISABLE=1` for box‑local NVLink systems.

- **Model downloads are slow**  
  Pre‑stage under `/data/lmdeploy/models` and point `MODEL_PATH` there. Keep `HF_HUB_ENABLE_HF_TRANSFER=0` unless you know you benefit from it.

---

## FAQ

**Q: Why not vLLM or SGLang?**  
**A:** Great projects, but on **Volta** the best performance/compatibility balance we observed came from **LMDeploy + TurboMind (FP16)**, especially for **long context**.

**Q: Can I quantize to run larger models?**  
**A:** Volta lacks BF16 and some modern quant kernels. You can try **int8** (`QUANT_POLICY=8`) at your own risk; expect speed/quality trade‑offs and test carefully.

**Q: How do I expose this behind a gateway?**  
**A:** Terminate TLS and auth at your API gateway (e.g., NGINX, APISIX, Traefik) and forward to `:23333`. The API is OpenAI‑compatible, so most clients work out‑of‑the‑box.

---

## Roadmap

- Add scripted **throughput benchmarking** and example **client notebooks**.
- Optional **prometheus exporter** for basic metrics.
- Example configs for **4‑GPU** and **2‑GPU** deployments.

---

## Acknowledgements

- **LMDeploy** — and the developers who continue to keep **Volta** usable.
- **Qwen** team for high‑quality 30B models with strong long‑context behavior.
- Community efforts around keeping **sm_70** systems productive.

---_Last updated: 2025-08-10_
