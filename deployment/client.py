#!/usr/bin/env python3
"""
client.py
---------
Command-line client for the Afro-FetalNet inference API.

Commands:
    predict   - classify a single image
    batch     - classify multiple images from a directory
    models    - list available / loaded models
    load      - load a model into the server
    unload    - unload a model from the server

Examples:
    python client.py predict --image samples/sample_frame.png --model efficientnet_b0
    python client.py batch   --dir ./images    --model efficientnet_b0 --output results.json
    python client.py models  --list-available
    python client.py load    --model efficientnet_b0 --version v1
"""

import argparse
import json
import sys
import socket
import threading
import queue
import time
import asyncio
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    pass

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' is not installed. Run: pip install requests")
    sys.exit(1)

try:
    import websockets
    import nest_asyncio
    nest_asyncio.apply()
    HAS_WS = True
except ImportError:
    HAS_WS = False

DEFAULT_BASE = "http://localhost:8000"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# Helpers

def _base(args) -> str:
    return args.base_url.rstrip("/")


def _check(resp):
    if not resp.ok:
        print(f"[HTTP {resp.status_code}] {resp.text}")
        sys.exit(1)
    return resp.json()


def _print_json(data):
    print(json.dumps(data, indent=2))


# Commands

def cmd_predict(args):
    path = Path(args.image)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    url = f"{_base(args)}/api/v1/inference/predict"
    params = {
        "model_name": args.model,
        "model_version": args.version,
        "top_k": args.top_k,
        "threshold": args.threshold,
    }
    with open(path, "rb") as f:
        resp = requests.post(url, params=params, files={"file": (path.name, f)})

    data = _check(resp)
    print(f"\n{path.name}  ->  {data['model_name']} ({data['model_version']})\n")
    for i, pred in enumerate(data["predictions"], 1):
        bar = "#" * int(pred["confidence"] * 30)
        print(f"  {i}. {pred['label']:<20} {pred['confidence']:.4f}  {bar}")
    print()


def cmd_batch(args):
    base = _base(args)
    dir_path = Path(args.dir)

    # Use server-side directory endpoint if server can access the path
    if args.server_side:
        url = f"{base}/api/v1/batch/predict-dir"
        params = {
            "directory": str(dir_path.resolve()),
            "model_name": args.model,
            "model_version": args.version,
            "top_k": args.top_k,
            "threshold": args.threshold,
            "recursive": args.recursive,
        }
        resp = requests.post(url, params=params)
        data = _check(resp)
    else:
        # Upload files from client machine
        image_paths = [
            p for p in dir_path.glob("**/*" if args.recursive else "*")
            if p.suffix.lower() in SUPPORTED
        ]
        if not image_paths:
            print("[WARN] No images found.")
            return

        print(f"Uploading {len(image_paths)} images...")
        url = f"{base}/api/v1/batch/predict"
        params = {
            "model_name": args.model,
            "model_version": args.version,
            "top_k": args.top_k,
            "threshold": args.threshold,
        }
        files = []
        handles = []
        for p in image_paths:
            fh = open(p, "rb")
            handles.append(fh)
            files.append(("files", (p.name, fh, "image/jpeg")))

        try:
            resp = requests.post(url, params=params, files=files)
            data = _check(resp)
        finally:
            for fh in handles:
                fh.close()

    # Print summary
    print(f"\n{data['total']} images processed\n")
    for r in data["results"][:20]:   # cap console output
        top = r["predictions"][0]["label"] if r["predictions"] else "-"
        conf = r["predictions"][0]["confidence"] if r["predictions"] else 0
        err = f"  [WARN] {r['error']}" if r.get("error") else ""
        print(f"  {r['filename']:<35} -> {top:<20} ({conf:.3f}){err}")

    if data["total"] > 20:
        print(f"  ... and {data['total'] - 20} more")

    # Save full results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(data["results"], f, indent=2)
        print(f"\nFull results saved to {args.output}")


def cmd_models(args):
    base = _base(args)
    if args.list_loaded or not args.list_available:
        data = _check(requests.get(f"{base}/api/v1/models/loaded"))
        print("\nLoaded models")
        for m in data["models"]:
            print(f"  {m['name']}:{m['version']}  [{m['device']}]")
        if not data["models"]:
            print("  (none)")

    if args.list_available:
        data = _check(requests.get(f"{base}/api/v1/models/available"))
        print("\nAvailable in registry")
        for m in data["models"]:
            status = "loaded" if m["loaded"] else "not loaded"
            print(f"  {m['name']}:{m['version']}  [{status}]")
        if not data["models"]:
            print("  (none - registry is empty)")
    print()


def cmd_load(args):
    base = _base(args)
    resp = requests.post(
        f"{base}/api/v1/models/load",
        json={"name": args.model, "version": args.version},
    )
    data = _check(resp)
    print(f"\nLoaded {data['name']}:{data['version']} on {data['device']}  ({data['num_classes']} classes)\n")


def cmd_unload(args):
    base = _base(args)
    resp = requests.post(
        f"{base}/api/v1/models/unload",
        json={"name": args.model, "version": args.version},
    )
    data = _check(resp)
    print(f"\nUnloaded {data['name']}:{data['version']}\n")


def cmd_stream(args):
    """
    Listen for UDP packets and display video with live, non-blocking inference overlays.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[ERROR] 'opencv-python' and 'numpy' are required for streaming.")
        sys.exit(1)

    # Shared state
    state = {
        "frame": None,
        "prediction": None,
        "running": True,
        "last_update": 0,
        "error": None,
        "lock": threading.Lock()
    }
    frame_q = queue.Queue(maxsize=1) 
    
    # Initial Check: Does the model exist and is it loaded?
    try:
        # Check available
        r = requests.get(f"{_base(args)}/api/v1/models/available", timeout=2.0)
        if r.ok:
            available = [m["name"] for m in r.json()["models"]]
            if args.model not in available:
                print(f"\n[WARNING] Model '{args.model}' not found in registry.")
                print(f"Available models: {', '.join(available)}\n")
        
        # Pre-load/Warmup (helps prevent initial timeout)
        print(f"Initializing/Warming up {args.model}:{args.version}...")
        requests.post(f"{_base(args)}/api/v1/models/load", 
                      json={"name": args.model, "version": args.version}, timeout=10.0)
    except Exception:
        print("[WARNING] Could not reach server to verify models. Inference may fail.")

    # Inference Worker
    def inference_worker():
        url = f"{_base(args)}/api/v1/inference/predict"
        params = {
            "model_name": args.model,
            "model_version": args.version,
            "top_k": 1,
            "threshold": args.threshold,
        }
        
        while state["running"]:
            try:
                # Get the latest frame from the queue, wait a bit
                f = frame_q.get(timeout=0.5)
                
                # Inference call
                _, encoded = cv2.imencode('.jpg', f)
                resp = requests.post(url, params=params, files={"file": ("frame.jpg", encoded.tobytes())}, timeout=10.0)
                
                with state["lock"]:
                    if resp.ok:
                        preds = resp.json()["predictions"]
                        state["prediction"] = preds[0] if preds else None
                        state["last_update"] = time.time()
                        state["error"] = None
                    else:
                        state["error"] = f"HTTP {resp.status_code}"
                        state["prediction"] = None
                
            except queue.Empty:
                continue
            except Exception as e:
                with state["lock"]:
                    state["error"] = "Inference Offline"
                    state["prediction"] = None
                time.sleep(1.0)

    # WebSocket Receiver (Relay Mode)
    async def ws_receiver():
        protocol = "wss" if args.base_url.startswith("https") else "ws"
        host = args.base_url.split("://")[-1].rstrip("/")
        ws_url = f"{protocol}://{host}/api/v1/inference/ws-stream"
        
        try:
            async with websockets.connect(ws_url) as ws:
                print(f"[INFO] Connected to relay server at {ws_url}")
                while state["running"]:
                    data = await ws.recv()
                    # WebSocket frames from our server are JPEG binary
                    arr = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        state["frame"] = frame
                        # Non-blocking push to inference
                        try:
                            if frame_q.full(): frame_q.get_nowait()
                            frame_q.put_nowait(frame.copy())
                        except: pass
        except Exception as e:
            print(f"[ERROR] Relay mode failed: {e}")
            state["running"] = False

    # UDP Receiver (Direct Mode)
    def udp_receiver():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("0.0.0.0", args.port))
            sock.settimeout(1.0)
            print(f"[INFO] Listening on direct UDP port {args.port}")
            while state["running"]:
                try:
                    data, _ = sock.recvfrom(65535)
                    arr = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        state["frame"] = frame
                        try:
                            if frame_q.full(): frame_q.get_nowait()
                            frame_q.put_nowait(frame.copy())
                        except: pass
                except socket.timeout:
                    continue
        except OSError as e:
            print(f"\n[ERROR] Port {args.port} is already in use by the server.")
            print("To fix: Run the client without --port or ensure 'run.py' is stopped.\n")
            state["running"] = False
        finally:
            sock.close()

    # Start Worker Threads
    inf_thread = threading.Thread(target=inference_worker, daemon=True)
    inf_thread.start()

    if HAS_WS:
        # Run WS receiver in an async-aware thread
        def run_ws():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(ws_receiver())
            except Exception as e:
                print(f"[ERROR] WebSocket Loop Error: {e}")
            finally:
                loop.close()
        recv_thread = threading.Thread(target=run_ws, daemon=True)
    else:
        print("[WARN] 'websockets' not installed. Falling back to direct UDP.")
        recv_thread = threading.Thread(target=udp_receiver, daemon=True)
    
    recv_thread.start()

    print(f"\nStream Receiver Active: model={args.model}:{args.version}")
    print("Press 'q' or 'ESC' to quit.\n")

    try:
        cv2.namedWindow("Fetal Ultrasound - Live Stream", cv2.WINDOW_AUTOSIZE)
        while state["running"]:
            with state["lock"]:
                frame = state["frame"]
                pred  = state["prediction"]
                err   = state["error"]
                
            if frame is not None:
                display = frame.copy()
                # Draw UI
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (350, 65), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
                
                if err:
                    cv2.putText(display, f"STATUS: {err}", (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)
                elif pred:
                    label, conf = pred['label'], pred['confidence']
                    color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
                    cv2.putText(display, f"PLANE: {label}", (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(display, f"CONF : {conf*100:.1f}%", (15, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
                else:
                    cv2.putText(display, "STATUS: ANALYZING...", (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1)

                cv2.imshow("Fetal Ultrasound - Live Stream", display)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            time.sleep(0.01)
    finally:
        state["running"] = False
        cv2.destroyAllWindows()


# Argument parser

def build_parser():
    parser = argparse.ArgumentParser(
        prog="client",
        description="Afro-FetalNet API client",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="API base URL")

    sub = parser.add_subparsers(dest="command", required=True)

    # predict
    p = sub.add_parser("predict", help="Classify a single image")
    p.add_argument("--image", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--version", default="latest")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.0)

    # batch
    p = sub.add_parser("batch", help="Classify images from a directory")
    p.add_argument("--dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--version", default="latest")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--output", default="", help="Save results to JSON file")
    p.add_argument("--recursive", action="store_true")
    p.add_argument(
        "--server-side",
        action="store_true",
        help="Use server-side directory endpoint (server must have access to the path)",
    )

    # models
    p = sub.add_parser("models", help="List models")
    p.add_argument("--list-available", action="store_true")
    p.add_argument("--list-loaded", action="store_true")

    # load
    p = sub.add_parser("load", help="Load a model into server memory")
    p.add_argument("--model", required=True)
    p.add_argument("--version", default="latest")

    # unload
    p = sub.add_parser("unload", help="Unload a model from server memory")
    p.add_argument("--model", required=True)
    p.add_argument("--version", default="latest")

    # stream
    p = sub.add_parser("stream", help="Receive live UDP stream and display predictions")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--model", required=True)
    p.add_argument("--version", default="latest")
    p.add_argument("--threshold", type=float, default=0.4)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "predict": cmd_predict,
        "batch": cmd_batch,
        "models": cmd_models,
        "load": cmd_load,
        "unload": cmd_unload,
        "stream": cmd_stream,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
