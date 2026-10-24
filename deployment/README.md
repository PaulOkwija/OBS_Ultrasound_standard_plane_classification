# Afro-FetalNet Inference Server

Offline inference for fetal ultrasound plane classifiers, built with FastAPI and PyTorch.
It serves models from a local, versioned registry and provides:

- a REST API for single-image and batch classification,
- a web UI for uploading images (`/`) and a live-video view (`/video`),
- a UDP bridge that receives frames from an ultrasound machine (simulated by `sender.py`),
- CLI tools for registering models, offline batch inference and talking to the API.

Runs on CPU; uses CUDA automatically when available. No internet access is needed at runtime.

## Quick start

```bash
cd deployment
pip install -r requirements.txt
cp .env.example .env          # optional; DEFAULT_MODEL=efficientnet_b0 preloads the bundled model
python run.py                 # or: uvicorn app.main:app --reload
```

| URL | What |
|---|---|
| http://localhost:8000 | Image upload UI |
| http://localhost:8000/video | Live / file video UI |
| http://localhost:8000/docs | Interactive API documentation |
| http://localhost:8000/health | Health check and loaded models |

Classify the bundled sample frame:

```bash
pip install -r requirements-dev.txt     # client, sender and test dependencies
python client.py predict --image samples/sample_frame.png --model efficientnet_b0
```

## Layout

```
deployment/
|-- app/
|   |-- main.py                 FastAPI app, startup/shutdown, static UI routes
|   |-- core/
|   |   |-- config.py           settings from environment / .env
|   |   |-- model_manager.py    registry discovery, version resolution, in-memory cache
|   |   `-- udp_bridge.py       UDP frame receiver, fans frames out to WebSocket clients
|   |-- models/loader.py        loads TorchScript or state_dict (timm, then torchvision)
|   |-- utils/
|   |   |-- preprocessing.py    image -> normalised tensor
|   |   `-- inference_engine.py forward pass, softmax, top-k
|   |-- api/routes/
|   |   |-- inference.py        POST /predict, WebSocket /ws-stream
|   |   |-- batch.py            POST /predict (multi-file), POST /predict-dir
|   |   `-- models.py           list / load / unload
|   `-- static/                 index.html, video.html
|-- model_registry/<name>/<version>/{model.pt, config.json}
|-- samples/sample_frame.png
|-- scripts/
|   |-- register_model.py       copy weights into the registry and write config.json
|   `-- batch_infer.py          offline batch inference without the server
|-- client.py                   CLI client
|-- sender.py                   UDP video sender (ultrasound simulator)
`-- tests/
```

## Registering a model

Each model version lives in `model_registry/<name>/<version>/` with a `model.pt` and a
`config.json`:

```json
{
  "format": "state_dict",
  "architecture": "efficientnet_b0",
  "num_classes": 3,
  "classes": ["brain", "femur", "abdomen"]
}
```

`format` is `state_dict` (architecture built with timm, falling back to torchvision) or
`torchscript` (architecture not needed). `classes` must be in the model's output order.
Checkpoints from `modelling/` use `["brain", "femur", "abdomen"]`.

Register a model soup produced by the modelling notebooks:

```bash
python scripts/register_model.py \
    --name efficientnet_b0_soup --version v1 \
    --weights ../modelling/checkpoints/averaged/efficientnet_b0_loco_averaged.pt \
    --architecture efficientnet_b0 --num-classes 3 \
    --classes brain femur abdomen --format state_dict
```

`latest` resolves to the highest version folder name (string order), so keep version names
like `v1`, `v2`, ... below `v10`, or zero-pad them.

## API

All inference endpoints take query parameters `model_name` (required), `model_version`
(default `latest`), `top_k` (default 5) and `threshold` (default 0.0). Models are loaded on
first use.

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/inference/predict` | Classify one uploaded image (`file`) |
| WS | `/api/v1/inference/ws-stream` | Stream of raw JPEG frames received over UDP |
| POST | `/api/v1/batch/predict` | Classify several uploaded images (`files`) |
| POST | `/api/v1/batch/predict-dir` | Classify every image in a directory on the server (`directory`, `recursive`) |
| GET | `/api/v1/models/available` | List registry contents |
| GET | `/api/v1/models/loaded` | List models in memory |
| POST | `/api/v1/models/load` | Body `{"name": ..., "version": ...}` |
| POST | `/api/v1/models/unload` | Body `{"name": ..., "version": ...}` |
| GET | `/health` | Status and loaded models |

Example (actual output for the bundled sample, a fetal head view):

```bash
curl -X POST "http://localhost:8000/api/v1/inference/predict?model_name=efficientnet_b0&top_k=3" \
     -F "file=@samples/sample_frame.png"
```

```json
{
  "model_name": "efficientnet_b0",
  "model_version": "v1",
  "predictions": [
    {"label": "brain", "class_index": 1, "confidence": 0.940794},
    {"label": "abdomen", "class_index": 0, "confidence": 0.04825},
    {"label": "femur", "class_index": 2, "confidence": 0.010956}
  ]
}
```

## CLI client

```bash
python client.py predict --image samples/sample_frame.png --model efficientnet_b0
python client.py batch   --dir ./images --model efficientnet_b0 --output results.json
python client.py models  --list-available
python client.py load    --model efficientnet_b0 --version v1
python client.py stream  --model efficientnet_b0 --port 5005
```

Use `--base-url` to point at a server on another machine.

## Offline batch inference (no server)

```bash
python scripts/batch_infer.py --model efficientnet_b0 --input ./images \
    --output results.json --top-k 3 --batch-size 32 --recursive
```

## Live video simulation

Simulates an ultrasound machine streaming to a tablet. Both ends can be on one machine.

1. On the receiving machine, start the server: `python run.py`. The UDP bridge listens on
   port 5005.
2. On the sending machine, stream a video file or a camera:
   ```bash
   python sender.py --video scan.mp4 --host <receiver-ip>     # --video 0 for a webcam
   ```
3. View the results either in the browser at `http://<receiver-ip>:8000/video` (select a
   model, switch the source to **Live**, press **Play**), or in a desktop window with
   `python client.py stream --model efficientnet_b0`.

Each frame is JPEG-encoded into a single UDP datagram, so it must stay under 65,507 bytes.
If frames are dropped, lower `--quality` or `--width` / `--height`. Both viewers decouple
display from inference and skip stale frames, so video stays smooth even when inference is
slower than the frame rate. Make sure the receiver's firewall allows UDP on port 5005.

## Configuration

Settings come from environment variables or `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `HOST`, `PORT` | `0.0.0.0`, `8000` | Bind address |
| `MODEL_REGISTRY_DIR` | `model_registry` | Registry path |
| `DEFAULT_MODEL` | empty | Model to load at startup (latest version) |
| `TOP_K` | `5` | Default number of predictions |
| `CONFIDENCE_THRESHOLD` | `0.0` | Minimum confidence returned |
| `IMAGE_SIZE` | `224` | Resize and crop size |
| `NORMALIZE_MEAN`, `NORMALIZE_STD` | ImageNet values | Normalisation |
| `MAX_BATCH_SIZE` | `64` | Images per forward pass in batch endpoints |

## Docker

```bash
docker compose up --build
DEFAULT_MODEL=efficientnet_b0 docker compose up
```

The registry is mounted read-only from `./model_registry`, port 8000 serves HTTP and port
5005/udp receives video frames.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
