import os, tempfile
import gradio as gr
from pydub import AudioSegment
from pyannote.audio import Pipeline
import soundfile as sf
import numpy as np

HF_TOKEN = os.getenv("HF_TOKEN")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

def diarize(audio):
    ata, sr = audio
              if not isinstance(data, np.ndarray):
        data = np.array(data)

        if data.ndim == 1:
        data = data[:, None]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, data, sr)
        tmp_path = tmp.name

    diarization = pipeline(tmp_path)
    original = AudioSegment.from_file(tmp_path)
    segments = {}
    text_out = ""

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        text_out += f"{turn.start:.1f}s - {turn.end:.1f}s : {speaker}\n"
        segment_audio = original[turn.start*1000 : turn.end*1000]
        segments.setdefault(speaker, AudioSegment.empty())
        segments[speaker] += segment_audio

    files = []
    for spk, audio_seg in segments.items():
        out_path = f"/tmp/{spk}.wav"
        audio_seg.export(out_path, format="wav")
        files.append(out_path)

    return text_out, files


demo = gr.Interface(
    diarize,
    inputs=gr.Audio(type="numpy", label="Upload áudio"),
    outputs=[
        gr.Textbox(label="Anotações"),
        gr.Files(label="Arquivos por locutor")
    ],
    title="Separação Voz Masculina/Feminina",
    description="Identifica locutores e gera um arquivo para cada um."
)

if __name__ == "__main__":
    demo.launch()
