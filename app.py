import os
import tempfile
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
    data, sr = audio

    if data is None or sr is None:
        raise ValueError("Áudio inválido ou não foi carregado corretamente.")

    data = np.asarray(data)

    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim == 2:
        if data.shape[1] == 2:
            data = np.mean(data, axis=1, keepdims=True)
        elif data.shape[1] > 2:
            raise ValueError("O áudio tem mais de 2 canais. Use áudio mono ou estéreo.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, data, sr)
        tmp_path = tmp.name

    diarization = pipeline(tmp_path)
    original = AudioSegment.from_file(tmp_path)
    segments = {}
    text_out = ""

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        text_out += f"{turn.start:.1f}s - {turn.end:.1f}s : {speaker}\n"
        segment_audio = original[int(turn.start * 1000): int(turn.end * 1000)]
        segments.setdefault(speaker, AudioSegment.empty())
        segments[speaker] += segment_audio

    audio_outputs = []
    for spk, audio_seg in segments.items():
        out_path = f"/tmp/{spk}.wav"
        audio_seg.export(out_path, format="wav")
        audio_outputs.append((f"Locutor {spk}", out_path))

    return text_out, audio_outputs

def create_demo():
    audio_outputs = []
    for i in range(5):  # até 5 locutores (ajustável)
        audio_outputs.append(gr.Audio(label=f"Locutor {i}", visible=False))

    return gr.Interface(
        diarize,
        inputs=gr.Audio(type="numpy", label="Upload áudio"),
        outputs=[
            gr.Textbox(label="Anotações"),
            *audio_outputs
        ],
        title="Separação Voz Masculina/Feminina",
        description="Identifica locutores e gera um player de áudio para cada um."
    )

demo = create_demo()

if __name__ == "__main__":
    demo.launch()
