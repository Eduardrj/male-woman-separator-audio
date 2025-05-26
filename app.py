import tempfile
import numpy as np
import soundfile as sf
import gradio as gr
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os

# Carrega o token de uma variável de ambiente (no Hugging Face Spaces, isso vem automaticamente)
auth_token = os.getenv("HUGGINGFACE_TOKEN")

if not auth_token:
    raise ValueError("Token de autenticação não encontrado. Certifique-se de adicionar o segredo 'HUGGINGFACE_TOKEN' nas configurações do seu espaço.")

try:
    # Tenta carregar o pipeline com o token de autenticação
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
    print("Pipeline carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o pipeline: {e}")

def diarize(audio):
    # Verifica se o pipeline foi carregado corretamente
    if not pipeline:
        raise ValueError("O pipeline não foi carregado corretamente. Verifique sua autenticação ou o modelo.")
    
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

# Interface Gradio
iface = gr.Interface(
    fn=diarize,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Audio(label="Locutor")],
    allow_flagging="never"
)

iface.launch()
