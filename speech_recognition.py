#   import torch
#
#   print(f"CUDA available? {torch.cuda.is_available()}")


# TRANSCRIPTION WORKS

import sounddevice as sd
import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
import sys
import time


def build_pipeline(model_id: str, torch_dtype: torch.dtype, device: str) -> Pipeline:
    """Creates a Hugging Face automatic-speech-recognition pipeline on the given device."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe


def record_audio(duration_seconds: int = 10) -> npt.NDArray:
    """Record duration_seconds of audio from default microphone.
    Return a single channel numpy array."""
    sample_rate = 16000  # Hz
    samples = int(duration_seconds * sample_rate)
    # Will use default microphone; on Jetson this is likely a USB WebCam
    audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype=np.float32)
    # Blocks until recording complete
    sd.wait()
    # Model expects single axis
    return np.squeeze(audio, axis=1)


if __name__ == "__main__":
    # Get model as argument, default to "distil-whisper/distil-medium.en" if not given
    model_id = sys.argv[1] if len(sys.argv) > 1 else "distil-whisper/distil-medium.en"
    print("Using model_id {model_id}")
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device {device}.")

    print("Building model pipeline...")
    pipe = build_pipeline(model_id, torch_dtype, device)
    print(type(pipe))
    print("Done")

    print("Recording...")
    audio = record_audio()
    print("Done")

    print("Transcribing...")
    start_time = time.time_ns()
    speech = pipe(audio)
    end_time = time.time_ns()
    print("Done")

    print(speech)
    print(f"Transcription took {(end_time-start_time)/1000000000} seconds")

    # TO DO: initialize the file for a warm start but ONLY record audio if GPIO is high

    # TO DO: Needs to pass the output text as a system argument to the prompt checkpoint (run the llm parse function used in "prompt_checkpoint")
