import gradio as gr
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import tempfile
import zipfile
import io # Para trabalhar com bytes em memória

# --- Configuração e Carregamento Inicial do Pipeline ---
auth_token = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = "pyannote/speaker-diarization" # Mantenha ou altere conforme sua necessidade

pipeline = None
pipeline_load_error = None

if not auth_token:
    pipeline_load_error = "ERRO CRÍTICO: Token de autenticação (HUGGINGFACE_TOKEN) não encontrado nas variáveis de ambiente."
    print(pipeline_load_error)
else:
    try:
        print(f"Tentando carregar o pipeline: {MODEL_NAME}...")
        pipeline = Pipeline.from_pretrained(MODEL_NAME, use_auth_token=auth_token)
        print("Pipeline carregado com sucesso!")
    except Exception as e:
        pipeline_load_error = (
            f"ERRO CRÍTICO AO CARREGAR O PIPELINE '{MODEL_NAME}': {e}\n"
            "Verifique se:\n"
            "1. O HUGGINGFACE_TOKEN é válido.\n"
            "2. Você aceitou os termos de uso do modelo (e suas dependências) no site do Hugging Face: "
            f"https://huggingface.co/{MODEL_NAME}"
        )
        print(pipeline_load_error)
# --- Fim do Carregamento Inicial ---

def diarize_and_create_zip(audio_filepath):
    if pipeline_load_error:
        return None, gr.update(value=pipeline_load_error, visible=True)
    if not pipeline:
        # Esta verificação é uma redundância se pipeline_load_error já foi definido, mas é uma boa prática.
        return None, gr.update(value="ERRO INESPERADO: Pipeline não está disponível, mas não houve erro de carregamento registrado.", visible=True)
    if not audio_filepath:
        return None, gr.update(value="Por favor, forneça um arquivo de áudio.", visible=True)

    try:
        print(f"Processando áudio: {audio_filepath}")
        diarization = pipeline(audio_filepath)

        audio_basename = os.path.basename(audio_filepath)
        diarization_summary = f"Resumo da Diarização para: {audio_basename}\n"
        diarization_summary += "---------------------------------------------------\n"

        # Carrega o áudio com pydub para segmentação
        try:
            audio_seg_full = AudioSegment.from_file(audio_filepath)
        except Exception as e_load:
            print(f"Erro ao carregar o arquivo de áudio com pydub: {e_load}")
            return None, gr.update(value=f"Erro ao ler o arquivo de áudio: {e_load}. Verifique o formato.", visible=True)

        speakers_segments = {} # Dicionário para armazenar segmentos de pydub por locutor

        if not diarization.get_timeline().support():
            diarization_summary += "Nenhum segmento de fala detectado.\n"
            return diarization_summary, None # Retorna o resumo e nenhum arquivo zip

        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            diarization_summary += f"De {turn.start:.2f}s até {turn.end:.2f}s: Locutor {speaker_label}\n"
            
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            
            # Garante que os tempos estão dentro dos limites do áudio
            start_ms = max(0, start_ms)
            end_ms = min(len(audio_seg_full), end_ms)

            if start_ms < end_ms: # Só adiciona se o segmento tiver duração
                segment = audio_seg_full[start_ms:end_ms]
                if speaker_label not in speakers_segments:
                    speakers_segments[speaker_label] = AudioSegment.empty()
                speakers_segments[speaker_label] += segment
        
        if not speakers_segments:
            diarization_summary += "Nenhum locutor com segmentos de fala válidos foi encontrado após o processamento.\n"
            return diarization_summary, None

        # Criar arquivo ZIP em memória
        zip_in_memory = io.BytesIO()
        with zipfile.ZipFile(zip_in_memory, 'w', zipfile.ZIP_DEFLATED) as zf:
            for speaker_label, full_segment in speakers_segments.items():
                if len(full_segment) > 0: # Exporta apenas se o segmento não for vazio
                    # Exportar o segmento pydub para um buffer de bytes em formato WAV
                    wav_buffer = io.BytesIO()
                    full_segment.export(wav_buffer, format="wav")
                    wav_buffer.seek(0) # Reposiciona o cursor para o início do buffer
                    zf.writestr(f"Locutor_{speaker_label}.wav", wav_buffer.read())
        
        zip_in_memory.seek(0)

        # Salvar o ZIP em memória para um arquivo temporário para Gradio poder servir
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip_file:
            tmp_zip_file.write(zip_in_memory.read())
            zip_filename_to_download = tmp_zip_file.name
        
        print(f"Arquivo ZIP gerado: {zip_filename_to_download}")
        return diarization_summary, gr.update(value=zip_filename_to_download, visible=True)

    except Exception as e:
        error_message = f"Erro durante a diarização: {e}"
        print(error_message)
        return None, gr.update(value=error_message, visible=True)


# --- Interface Gradio ---
with gr.Blocks(title="Diarização de Áudio Avançada") as iface:
    gr.Markdown(
        """
        # Diarização de Áudio com `pyannote.audio`
        Faça o upload de um arquivo de áudio (MP3, WAV, etc.) para identificar os diferentes locutores,
        ver quando cada um falou e baixar os áudios separados em um arquivo ZIP.
        """
    )

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Faça Upload do Seu Arquivo de Áudio")
    
    analyze_button = gr.Button("Analisar Áudio e Gerar ZIP", variant="primary")
    
    with gr.Column():
        info_textbox = gr.Textbox(label="Informações da Diarização / Status", lines=10, interactive=False)
        zip_output_file = gr.File(label="Baixar Áudios dos Locutores (Arquivo ZIP)", visible=False) # Começa invisível

    # Se o pipeline falhou ao carregar na inicialização, exibe a mensagem de erro e desabilita o botão.
    if pipeline_load_error:
        info_textbox.value = pipeline_load_error # Define o valor inicial
        analyze_button.interactive = False # Desabilita o botão
        gr.Markdown(f"<h3 style='color:red;'>Falha Crítica ao Carregar o Modelo de IA. Verifique a mensagem acima.</h3>")

    analyze_button.click(
        fn=diarize_and_create_zip,
        inputs=audio_input,
        outputs=[info_textbox, zip_output_file] # Mapeia para os dois componentes de saída
    )

    gr.Markdown(
        """
        **Como funciona:**
        1. O áudio é processado para identificar diferentes locutores e os momentos em que falam.
        2. Um resumo textual da diarização é exibido.
        3. Um arquivo ZIP é gerado contendo os segmentos de áudio de cada locutor como arquivos WAV separados.
        
        **Observações:**
        - O processamento pode levar alguns minutos para áudios longos.
        - A qualidade da diarização depende da clareza do áudio e da distinção entre as vozes.
        - Certifique-se de que o token `HUGGINGFACE_TOKEN` está configurado corretamente no ambiente se o modelo for privado/gated.
        """
    )

if __name__ == "__main__":
    iface.launch()