#!/usr/bin/env python3
"""
sender.py
---------
Simulates an ultrasound machine by streaming video frames over UDP to the
inference server (app/core/udp_bridge.py listens on port 5005 by default).

Each frame is resized, JPEG-encoded and sent as a single UDP datagram, so the
encoded frame must stay under the 65,507-byte UDP payload limit. Lower
--quality or --width/--height if frames are dropped.

Usage:
    python sender.py --video scan.mp4 --host 192.168.1.20
    python sender.py --video 0 --host 127.0.0.1          # webcam
"""

import argparse
import socket
import time

import cv2

MAX_UDP_PAYLOAD = 65507


def parse_args():
    parser = argparse.ArgumentParser(description="Stream video frames over UDP")
    parser.add_argument("--video", default="video.mp4",
                        help="Path to a video file, or a camera index such as 0")
    parser.add_argument("--host", default="127.0.0.1", help="IP address of the receiving server")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality, 0 to 100")
    parser.add_argument("--fps", type=float, default=30.0, help="Target send rate")
    return parser.parse_args()


def main():
    args = parse_args()

    # Step 1: open the video source (a digit means a camera index).
    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.video}")

    # Step 2: open a UDP socket pointed at the receiver.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
    frame_interval = 1.0 / args.fps
    print(f"Streaming UDP to {args.host}:{args.port} (Ctrl+C to stop)")

    # Step 3: read, compress and send frames, looping the video at the end.
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (args.width, args.height))
            _, encoded = cv2.imencode(".jpg", frame, encode_params)

            if len(encoded) < MAX_UDP_PAYLOAD:
                sock.sendto(encoded.tobytes(), (args.host, args.port))
            else:
                print(f"Frame dropped: {len(encoded)} bytes exceeds the UDP limit.")

            time.sleep(frame_interval)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        sock.close()


if __name__ == "__main__":
    main()
