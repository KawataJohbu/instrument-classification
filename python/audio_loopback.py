"""Windowsの再生先をWASAPIループバックで取得する（マイクは使わない）。"""
from __future__ import annotations

import math
import sys
import threading
import time


def backend():
    if sys.platform != "win32":
        raise RuntimeError("Loopback input is supported on Windows only.")
    try:
        import pyaudiowpatch
    except ImportError as error:
        raise RuntimeError("Install loopback support: python -m pip install PyAudioWPatch") from error
    return pyaudiowpatch


def list_devices():
    api = backend()
    with api.PyAudio() as audio:
        default = audio.get_default_wasapi_loopback()
        return [dict(info, selected=info["index"] == default["index"])
                for info in audio.get_loopback_device_info_generator()]


def record(seconds, device=None, stop=None):
    """モノラルの16bit相当の値と実際のサンプルレートを返す。

    デバイス固有のレートで録音して不要な変換を避ける。
    再生停止時にコールバックが来なくても、実時間で待ち時間を制限する。
    """
    import numpy as np

    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Recording seconds must be positive and finite.")
    api = backend()
    with api.PyAudio() as audio:
        info = (audio.get_default_wasapi_loopback() if device is None
                else audio.get_wasapi_loopback_analogue_by_index(device))
        channels = int(info["maxInputChannels"])
        rate = int(info["defaultSampleRate"])
        if channels < 1 or rate < 1 or not info.get("isLoopbackDevice"):
            raise ValueError("Select a WASAPI loopback device from --list-loopback-devices.")
        frames = max(1, round(seconds*rate))
        chunks = []
        received = 0
        completed = threading.Event()
        overflow = []

        def callback(data, count, timing, status):
            nonlocal received
            if stop is not None and stop.is_set():
                completed.set()
                return (None, api.paComplete)
            if status & api.paInputOverflow:
                overflow.append(status)
            chunks.append(data)
            received += len(data)//(4*channels)
            if received >= frames:
                completed.set()
                return (None, api.paComplete)
            return (None, api.paContinue)

        with audio.open(format=api.paFloat32, channels=channels, rate=rate,
                        input=True, input_device_index=int(info["index"]),
                        frames_per_buffer=512, stream_callback=callback):
            deadline = time.monotonic()+seconds+0.5
            while not completed.wait(0.05):
                if (stop is not None and stop.is_set()) or time.monotonic() >= deadline:
                    break
        if overflow:
            raise RuntimeError("Loopback audio overflowed. Close heavy applications and retry.")
        if stop is not None and stop.is_set():
            return [], rate
        samples = np.frombuffer(b"".join(chunks), dtype="<f4").reshape(-1, channels)
        mono = samples[:frames].mean(axis=1) * 32768.0
        # 無再生中は不足分を無音として扱い、前回の音を使い回さない。
        if len(mono) < frames:
            mono = np.pad(mono, (0, frames-len(mono)))
        return mono.tolist(), rate
