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
import requests

from ollama import Client

from Jetson import GPIO as GPIO
import time

# GPIO.cleanup()
# Use physical pin numbering
GPIO.setmode(GPIO.BOARD)

# Pin 7 = GPIO09
button_pin = 7

# Set pin as input with pull-down
GPIO.setup(button_pin, GPIO.IN)


LLM_MODEL: str = "gemma3:27b"  # Optional, change this to be the model you want
client: Client = Client(
    host="http://10.1.69.213:11434"  # Optional, change this to be the URL of your LLM
)


def llm_parse_for_wttr(user_prompt):

    # prompt = sys.argv[1] # first argument after filename.py
    response = client.chat(  # from terminal to the LLM "prompt" variable
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": transcribed_text},
            {
                "role": "system",
                "content": """

          Return your answer in one of four formats. The first format will be to reformatting the city name provided to replace spaces with a +. In this case the user would say something
          like: Can I get the weather from Los angeles? You would return Los+Angeles
          
          The second format will be if the user provides a landmark instead of a city name in their weather request. In this case you would receive a weather 
          request that contains the name of a landmark. For example: "What is the weather at the Eiffel Tower?"

          you would return the name of the landmark with a tilda in front of it and then replace spaces with a plus sign. For this example you would return ~Eiffel+Tower

          The third format will be if the user asks for the weather at an airport, and in this case you would return the three letter airport identifier.
          identifier code, in this case, you would recieve something like "Hey, what's the weather at Denver Airport" and you would return dia
      
          Here is a function that may help:

          # Example dummy formatter function
          def formatter(input_text):
              # Logic to reformat input_text to whatever you need
              return input_text.replace(" ", "+")  # simple example

              output = formatter(input_text)

        """,
            },
        ],
        # model='gemma3:27b',
    )

    return response["message"]["content"]


# end of prompt stuff


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


def record_audio(duration_seconds: int = 5) -> npt.NDArray:
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


# """This script evaluates an LLM prompt for processing text so that it can be used for the wttr.in API"""
# import sys
# from ollama import Client

# LLM_MODEL: str = "gemma3:27b"    # Optional, change this to be the model you want
# client: Client = Client(
#   host='http://10.1.69.21:11434' # Optional, change this to be the URL of your LLM 10.1.69.213:11434  USED TO BE : http://ai.dfec.xyz:11434
# )

# # TODO: define  llm_parse_for_wttr()

# def llm_parse_for_wttr(transcribed_text: str):

#   #prompt = sys.argv[1] # first argument after filename.py
#   response = client.chat( # from terminal to the LLM "prompt" variable
#     model=LLM_MODEL,
#     messages=[
#       {'role': 'user', 'content': transcribed_text},

#       {'role': 'system', 'content': '''

#           Return your answer in one of four formats. The first format will be to reformatting the city name provided to replace spaces with a +. In this case the user would say something
#           like: Can I get the weather from Los angeles? You would return Los+Angeles

#           The second format will be if the user provides a landmark instead of a city name in their weather request. In this case you would receive a weather
#           request that contains the name of a landmark. For example: "What is the weather at the Eiffel Tower?"

#           you would return the name of the landmark with a tilda in front of it and then replace spaces with a plus sign. For this example you would return ~Eiffel+Tower

#           The third format will be if the user asks for the weather at an airport, and in this case you would return the three letter airport identifier.
#           identifier code, in this case, you would recieve something like "Hey, what's the weather at Denver Airport" and you would return dia

#           Never return the actual weather at that location because this will be dealt with later.

#           Here is a function that may help:

#           # Example dummy formatter function
#           def formatter(input_text):
#               # Logic to reformat input_text to whatever you need
#               return input_text.replace(" ", "+")  # simple example

#               output = formatter(input_text)

#         '''
#       }

#     ],
#                      #model='gemma3:27b',
# )

#   return response['message']['content']


def create_get_request(parse_output) -> str:  # ollama is used to reformat
    # passes transcription to the model, then the rer\formatted city name becomes the result

    # URL to send the get request to
    response = requests.get(f"https://wttr.in/{parse_output}")

    return response.text

    # return transcription_output


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

    # button
    while True:
        print("waiting for button...")
        GPIO.wait_for_edge(button_pin, GPIO.RISING)
        print("Button Pressed!")

        print("Recording...")
        audio = record_audio()
        print("Done")

        print("Transcribing...")
        start_time = time.time_ns()
        speech = pipe(audio)  # speech is the transcribed stuff
        transcribed_text = speech["text"]
        end_time = time.time_ns()
        print("Done")

        print(speech)
        print(f"Transcription took {(end_time-start_time)/1000000000} seconds")

        # pass speech as an input into a function
        # TO DO: Needs to pass the output text as a system argument to the
        # prompt checkpoint
        # (run the llm parse function used in "prompt_checkpoint")

        parse_result = llm_parse_for_wttr(transcribed_text)
        print(f"LLM returned: {parse_result}")

        content = create_get_request(
            parse_result
        )  # this function prints the result automatically

        print(content)
