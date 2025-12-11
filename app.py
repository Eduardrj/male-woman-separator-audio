import gradio as gr
# Compat: torchaudio API change (set_audio_backend removed in newer versions)
try:
    import torchaudio  # noqa: F401
    if not hasattr(torchaudio, "set_audio_backend"):
        def _noop_set_audio_backend(_name: str):
            # Newer torchaudio uses default backends and no longer exposes this setter.
            # Pyannote 3.x calls this; a no-op preserves compatibility.
            return None
        setattr(torchaudio, "set_audio_backend", _noop_set_audio_backend)
except Exception:
    # If torchaudio is unavailable, pyannote may still work via soundfile fallback.
    pass

from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import tempfile
import zipfile
import io # Para trabalhar com bytes em memória
import time # Para medir o tempo de processamento

# --- Configuração e Carregamento Inicial do Pipeline ---
# Tenta obter o token das variáveis de ambiente (ideal para Hugging Face Spaces)
auth_token = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = "pyannote/speaker-diarization" # Mantenha ou altere conforme sua necessidade (ex: "pyannote/speaker-diarization-pytorch")

pipeline = None
pipeline_load_error = None # Armazena mensagem de erro se o pipeline falhar ao carregar

if not auth_token:
    pipeline_load_error = "ERRO CRÍTICO: Token de autenticação (HUGGINGFACE_TOKEN) não encontrado nas variáveis de ambiente."
    print(pipeline_load_error)
else:
    try:
        print(f"Tentando carregar o pipeline: {MODEL_NAME}...")
        # Tenta carregar o pipeline
        temp_pipeline = Pipeline.from_pretrained(MODEL_NAME, use_auth_token=auth_token)
        
        # Verifica se o pipeline foi realmente carregado e não é None
        if temp_pipeline:
            pipeline = temp_pipeline
            print(f"Pipeline '{MODEL_NAME}' parece ter sido carregado com sucesso.")
            # Você poderia adicionar um pequeno teste aqui se a pyannote tiver um método de verificação,
            # mas geralmente, se não houver exceção e não for None, está ok para prosseguir.
        else:
            # Caso a pyannote imprima um erro interno e retorne None sem lançar uma exceção
            pipeline_load_error = (
                f"Falha ao carregar o pipeline '{MODEL_NAME}'. A pyannote pode ter retornado None. "
                "Verifique os logs da pyannote acima e os termos de uso do modelo no Hugging Face."
            )
            print(pipeline_load_error)
            
    except Exception as e:
        pipeline = None # Garante que é None em caso de exceção
        pipeline_load_error = (
            f"EXCEÇÃO CRÍTICA AO CARREGAR O PIPELINE '{MODEL_NAME}': {e}\n"
            "Verifique se:\n"
            "1. O HUGGINGFACE_TOKEN é válido.\n"
            "2. Você aceitou os termos de uso do modelo (e suas dependências, se houver) no site do Hugging Face: "
            f"https://huggingface.co/{MODEL_NAME}\n"
            "3. Há conexão com a internet."
        )
        print(pipeline_load_error)
# --- Fim do Carregamento Inicial ---

def diarize_and_create_zip(audio_filepath):
    # Registra o tempo de início do processamento da função
    func_process_start_time = time.time()

    # Verificações iniciais de erro (pipeline não carregado na inicialização do app)
    if pipeline_load_error:
        return gr.update(value=pipeline_load_error, visible=True), None
    
    # Verificação adicional caso pipeline_load_error não tenha sido definido, mas pipeline ainda é None
    if not pipeline:
        # Calcula o tempo até este ponto se necessário, mas o erro principal é o pipeline
        processing_duration_note = f"\n(Tempo decorrido até a detecção do erro: {time.time() - func_process_start_time:.2f}s)"
        return gr.update(value="ERRO INESPERADO: Pipeline não está disponível (não carregado corretamente na inicialização)." + processing_duration_note, visible=True), None
    
    if not audio_filepath:
        processing_duration_note = f"\n(Tempo decorrido: {time.time() - func_process_start_time:.2f}s)"
        return gr.update(value="Por favor, forneça um arquivo de áudio." + processing_duration_note, visible=True), None

    processing_duration_note = "" # Será preenchido no final ou em caso de erro
    zip_filename_to_download = None # Inicializa

    try:
        print(f"Processando áudio: {audio_filepath}")
        diarization = pipeline(audio_filepath)

        audio_basename_for_summary = os.path.basename(audio_filepath)
        diarization_summary = f"Resumo da Diarização para: {audio_basename_for_summary}\n"
        diarization_summary += "---------------------------------------------------\n"

        try:
            audio_seg_full = AudioSegment.from_file(audio_filepath)
        except Exception as e_load:
            print(f"Erro ao carregar o arquivo de áudio com pydub: {e_load}")
            func_process_end_time = time.time()
            processing_duration_seconds = func_process_end_time - func_process_start_time
            error_msg = f"Erro ao ler o arquivo de áudio: {e_load}. Verifique o formato.\n(Processamento levou {processing_duration_seconds:.2f}s até o erro)."
            return gr.update(value=error_msg, visible=True), None

        speakers_segments = {} 

        if not diarization.get_timeline().support():
            diarization_summary += "Nenhum segmento de fala detectado.\n"
        else:
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                diarization_summary += f"De {turn.start:.2f}s até {turn.end:.2f}s: Locutor {speaker_label}\n"
                
                start_ms = int(turn.start * 1000)
                end_ms = int(turn.end * 1000)
                start_ms = max(0, start_ms)
                end_ms = min(len(audio_seg_full), end_ms)

                if start_ms < end_ms: 
                    segment = audio_seg_full[start_ms:end_ms]
                    if speaker_label not in speakers_segments:
                        speakers_segments[speaker_label] = AudioSegment.empty()
                    speakers_segments[speaker_label] += segment
            
        if not speakers_segments and diarization.get_timeline().support():
            diarization_summary += "Nenhum locutor com segmentos de fala válidos foi encontrado após o processamento.\n"
        
        if speakers_segments: # Só cria ZIP se houver segmentos de locutores
            zip_in_memory = io.BytesIO()
            with zipfile.ZipFile(zip_in_memory, 'w', zipfile.ZIP_DEFLATED) as zf:
                for speaker_label, full_segment in speakers_segments.items():
                    if len(full_segment) > 0: 
                        wav_buffer = io.BytesIO()
                        full_segment.export(wav_buffer, format="wav")
                        wav_buffer.seek(0) 
                        zf.writestr(f"Locutor_{speaker_label}.wav", wav_buffer.read())
            
            zip_in_memory.seek(0)

            # --- LÓGICA PARA NOMEAR O ARQUIVO ZIP ---
            audio_input_basename_sem_ext = os.path.splitext(os.path.basename(audio_filepath))[0]
            desired_zip_filename = f"{audio_input_basename_sem_ext}.zip"
            temp_dir = tempfile.gettempdir()
            final_zip_path_on_disk = os.path.join(temp_dir, desired_zip_filename)
            
            try:
                with open(final_zip_path_on_disk, 'wb') as f_out:
                    f_out.write(zip_in_memory.read())
                zip_filename_to_download = final_zip_path_on_disk
                print(f"Arquivo ZIP nomeado gerado em: {zip_filename_to_download}")
            except Exception as e_write_zip:
                print(f"Erro ao salvar o arquivo ZIP nomeado ({final_zip_path_on_disk}): {e_write_zip}")
                # Fallback: usa um nome temporário aleatório
                try:
                    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip_file:
                        # É preciso rebobinar zip_in_memory se read() já foi chamado e falhou na escrita anterior
                        zip_in_memory.seek(0) 
                        tmp_zip_file.write(zip_in_memory.read())
                        zip_filename_to_download = tmp_zip_file.name
                    print(f"Arquivo ZIP de fallback (nome aleatório) gerado: {zip_filename_to_download}")
                except Exception as e_fallback_zip:
                    print(f"Erro ao criar ZIP de fallback: {e_fallback_zip}")
                    zip_filename_to_download = None # Garante que é None
            # --- FIM DA LÓGICA PARA NOMEAR O ARQUIVO ZIP ---

        # Registra o tempo de fim do processamento e calcula a duração
        func_process_end_time = time.time()
        processing_duration_seconds = func_process_end_time - func_process_start_time
        processing_duration_note = f"\n---------------------------------------------------\nTempo total de processamento desta análise: {processing_duration_seconds:.2f} segundos."
        
        diarization_summary += processing_duration_note
        
        return gr.update(value=diarization_summary, visible=True), gr.update(value=zip_filename_to_download, visible=True if zip_filename_to_download else False)

    except Exception as e:
        func_process_end_time = time.time()
        processing_duration_seconds = func_process_end_time - func_process_start_time
        
        error_message = f"Erro durante a diarização: {e}"
        print(error_message)
        processing_duration_note = f"\n(Processamento levou {processing_duration_seconds:.2f}s até o erro)."
        return gr.update(value=error_message + processing_duration_note, visible=True), None


# --- Interface Gradio ---
with gr.Blocks(title="Diarização de Áudio Avançada") as iface:
    gr.Markdown(
        """
        # Diarização de Áudio com `pyannote.audio`
        Faça o upload de um arquivo de áudio (MP3, WAV, etc.) para identificar os diferentes locutores,
        ver quando cada um falou e baixar os áudios separados em um arquivo ZIP.
        O nome do arquivo ZIP será o mesmo do arquivo de áudio enviado.
        """
    )

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Faça Upload do Seu Arquivo de Áudio")
    
    analyze_button = gr.Button("Analisar Áudio e Gerar ZIP", variant="primary")
    
    with gr.Column():
        info_textbox = gr.Textbox(label="Informações da Diarização / Status do Processamento", lines=12, interactive=False)
        zip_output_file = gr.File(label="Baixar Áudios dos Locutores (Arquivo ZIP)", visible=False) 

    # Se o pipeline falhou ao carregar na inicialização, exibe a mensagem de erro e opcionalmente desabilita o botão.
    if pipeline_load_error:
        # Define o valor inicial do textbox de informação se houver erro no carregamento do pipeline
        # Esta atribuição direta pode não funcionar como esperado para atualizar um componente Gradio
        # após a definição do layout. A atualização via `gr.update` em uma função é mais confiável.
        # No entanto, para uma mensagem inicial, podemos tentar definir aqui ou usar um gr.HTML/Markdown.
        # Para simplificar, o erro será mostrado quando o botão for clicado, se `pipeline_load_error` estiver definido.
        # Ou, podemos adicionar um Markdown estático para o erro.
        gr.Markdown(f"<h3 style='color:red;'>AVISO NA INICIALIZAÇÃO: {pipeline_load_error}</h3>")
        # Se quiser desabilitar o botão se o pipeline falhar ao carregar:
        # analyze_button.interactive = False (Isso precisaria estar em um evento de carregamento do Gradio para funcionar dinamicamente)
        # Por enquanto, a verificação no início da função `diarize_and_create_zip` lidará com isso.

    analyze_button.click(
        fn=diarize_and_create_zip,
        inputs=audio_input,
        outputs=[info_textbox, zip_output_file] 
    )

    gr.Markdown(
        """
        **Como funciona:**
        1. O áudio é processado para identificar diferentes locutores e os momentos em que falam.
        2. Um resumo textual da diarização, incluindo o tempo de processamento, é exibido.
        3. Um arquivo ZIP é gerado contendo os segmentos de áudio de cada locutor como arquivos WAV separados.
        
        **Observações:**
        - O processamento pode levar alguns minutos para áudios longos.
        - A qualidade da diarização depende da clareza do áudio e da distinção entre as vozes.
        - Certifique-se de que o token `HUGGINGFACE_TOKEN` está configurado corretamente no ambiente se o modelo for privado/gated.
        """
    )

if __name__ == "__main__":
    # Imprime o status do pipeline ao iniciar o app (no console)
    if pipeline_load_error:
        print(f"STATUS DA INICIALIZAÇÃO DO PIPELINE: FALHOU. Erro: {pipeline_load_error}")
    elif pipeline:
        print("STATUS DA INICIALIZAÇÃO DO PIPELINE: CARREGADO COM SUCESSO.")
    else:
        print("STATUS DA INICIALIZAÇÃO DO PIPELINE: NÃO CARREGADO (causa desconhecida ou token não fornecido).")
    
    iface.launch()