---
title: Male Woman Separator Audio
emoji: 📚
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
python: 3.10
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Configuração do Space
- Defina o segredo `HUGGINGFACE_TOKEN` em Settings > Secrets do Space (precisa ter acesso ao modelo `pyannote/speaker-diarization`).
- Verifique se os termos do modelo foram aceitos.
- Python: `3.10` (definido acima) é compatível com `torch/torchaudio` e `gradio 5.31.0`.

## Teste local (Windows)
```powershell
# No diretório do projeto
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
