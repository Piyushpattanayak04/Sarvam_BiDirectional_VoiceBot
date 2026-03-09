"""Full pipeline test: TTS-generated speech → WebSocket → VAD → STT → NLU → Dialogue → TTS response"""
import asyncio
import base64
import io
import json
import wave

import websockets
from sarvamai import SarvamAI


async def test_real_speech():
    # Generate real speech audio first
    print("Generating speech audio with TTS...")
    client = SarvamAI(api_subscription_key="sk_fo96ffip_JuivmZvViFkcN1vMBMuNFNLN")
    tts_resp = client.text_to_speech.convert(
        text="I want to block my debit card",
        target_language_code="en-IN",
        model="bulbul:v3",
        speaker="shubh",
    )
    wav_bytes = base64.b64decode(tts_resp.audios[0])

    # Extract raw PCM from WAV
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
        print(f"  WAV: {sr}Hz, {wf.getnframes()} frames, {len(pcm)} bytes PCM")

    # Connect
    uri = "ws://localhost:8001/ws/voice"
    async with websockets.connect(uri) as ws:
        print("Connected")

        # Drain greeting
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=8.0)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    msg_type = data.get("type", "?")
                    print(f"  Greeting: {msg_type}")
                else:
                    print(f"  Greeting audio: {len(msg)} bytes")
        except asyncio.TimeoutError:
            print("Greeting drained")

        # Send speech as 20ms chunks (320 samples = 640 bytes at 16kHz)
        CHUNK_BYTES = 640
        total_chunks = len(pcm) // CHUNK_BYTES
        print(f"Sending {total_chunks} chunks of speech audio...")
        for i in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[i : i + CHUNK_BYTES]
            if len(chunk) == CHUNK_BYTES:
                await ws.send(chunk)
                await asyncio.sleep(0.015)

        # Send 2s silence to trigger speech end
        print("Sending 2s silence...")
        silence = bytes(CHUNK_BYTES)
        for _ in range(100):
            await ws.send(silence)
            await asyncio.sleep(0.02)

        # Wait for response
        print("Waiting for response...")
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=20.0)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    print(f"  Response: {json.dumps(data)}")
                else:
                    print(f"  Response audio: {len(msg)} bytes")
        except asyncio.TimeoutError:
            print("Done waiting")


asyncio.run(test_real_speech())
