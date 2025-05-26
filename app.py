import tempfile
import numpy as np
import soundfile as sf
import gradio as gr
from pyannote.audio import Pipeline
from pydub import AudioSegment

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

def diarize(audio):
    diarization = pipeline(audio)

    audio_seg = AudioSegment.from_file(audio)
    sr = audio_seg.frame_rate
    samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32) / (1 << (8 * audio_seg.sample_width - 1))

    speakers = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = AudioSegment.silent(duration=0)

        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        speakers[speaker] += audio_seg[start_ms:end_ms]

    outputs = []
    for speaker, segment in speakers.items():
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            samples = np.array(segment.get_array_of_samples()).astype(np.float32) / (1 << (8 * segment.sample_width - 1))
            samples = samples if samples.ndim > 1 else samples[:, None]  # Corrige áudio mono
            sf.write(tmp.name, samples, sr)
            outputs.append((f"Locutor: {speaker}", tmp.name))

    return outputs

iface = gr.Interface(
    fn=diarize,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Audio(label="Locutor")],
    allow_flagging="never"
)

iface.launch()
