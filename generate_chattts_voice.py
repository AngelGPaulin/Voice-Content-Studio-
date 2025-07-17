import torch
import torchaudio
import ChatTTS
import os
import numpy as np

# --- Configuración Principal ---
# Cambia esta variable para seleccionar el orador deseado:
# "hombre" para la voz masculina (speaker_hombre_7_emb.pt)
# "mujer" para la voz femenina (speaker_mujer_10_emb.pt)
USE_SPEAKER = "mujer" # <--- CAMBIA ESTO A "hombre" O "mujer" SEGÚN TU PREFERENCIA

output_base_filename = "reddit_story_chattts_voice"

# --- Nueva historia de Reddit en inglés ---
reddit_story_text = """
Okay Reddit, I need some advice. My cat, Whiskers, has started bringing me 'gifts' every morning. Not dead mice, thankfully, but things like my car keys, a single sock, or even once, my neighbor's garden gnome! It's sweet, but also incredibly inconvenient. How do I gently tell him that while I appreciate the thought, I'm perfectly capable of finding my own car keys?
"""

# --- Inicialización y Generación de Voz ---
print("Inicializando ChatTTS...")

# Determinar si usar GPU (cuda) o CPU
if torch.cuda.is_available():
    print("GPU (CUDA) detectada. ChatTTS usará la GPU.")
    torch_device = "cuda"
else:
    print("GPU (CUDA) NO detectada. ChatTTS usará la CPU. Esto será muy lento.")
    torch_device = "cpu"

try:
    # Cargar el modelo de ChatTTS usando ChatTTS.Chat()
    chat = ChatTTS.Chat()
    
    # Cargar el modelo. compile=False para evitar problemas iniciales de compilación.
    chat.load(compile=False, device=torch_device) 

    print("Modelo de ChatTTS cargado exitosamente.")

    # --- Cargar Hablante Seleccionado ---
    ruta_speaker_hombre = "speaker_hombre.pt"
    ruta_speaker_mujer = "speaker_mujer.pt"

    selected_spk_emb = None
    output_audio_name = ""

    if USE_SPEAKER == "hombre":
        print(f"\nCargando hablante masculino desde: {ruta_speaker_hombre}")
        selected_spk_emb = torch.load(ruta_speaker_hombre)
        output_audio_name = f"{output_base_filename}_masculino.wav"
    elif USE_SPEAKER == "mujer":
        print(f"\nCargando hablante femenino desde: {ruta_speaker_mujer}")
        selected_spk_emb = torch.load(ruta_speaker_mujer)
        output_audio_name = f"{output_base_filename}_femenino.wav"
    else:
        raise ValueError("La variable USE_SPEAKER debe ser 'hombre' o 'mujer'.")

    if selected_spk_emb is None:
        raise ValueError("No se pudo cargar la incrustación del hablante. ¡Verifica la ruta y el valor de USE_SPEAKER!")

    print(f"\nGenerando audio con el hablante seleccionado ({USE_SPEAKER})...")
        
    # Configurar los parámetros de inferencia de código
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=selected_spk_emb, # Usamos la incrustación de hablante cargada
        temperature=0.3,
        top_P=0.7,
        top_K=20,
    )

    # Configurar los parámetros de refinamiento de texto (puedes ajustarlos)
    # '[oral_2][laugh_0][break_6]' es un buen punto de partida para "amigable"
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_2][laugh_0][break_6]', 
    )

    # Llamar a infer para generar el audio
    # `texts` debe ser una lista, incluso si es un solo texto.
    wavs = chat.infer(
        [reddit_story_text], # El texto de la historia de Reddit
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
    )
    
    # `wavs` es una lista, tomamos el primer elemento.
    generated_wav = wavs[0] 

    # Guardar el archivo de audio
    output_filepath = os.path.join(os.getcwd(), output_audio_name) 
    
    if isinstance(generated_wav, torch.Tensor):
        audio_tensor = generated_wav
    elif isinstance(generated_wav, (list, tuple)):
        audio_tensor = torch.from_numpy(np.array(generated_wav)).float()
    elif isinstance(generated_wav, np.ndarray):
        audio_tensor = torch.from_numpy(generated_wav).float()
    else:
        raise TypeError(f"Tipo de dato de audio no soportado: {type(generated_wav)}")

    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    if audio_tensor.dtype != torch.float32:
        audio_tensor = audio_tensor.float()
    
    try:
        max_val = audio_tensor.abs().max()
        if max_val > 1.0:
            audio_tensor = audio_tensor / max_val
    except Exception as e:
        print(f"Advertencia: No se pudo normalizar el audio para {output_audio_name}. Error: {e}")

    torchaudio.save(output_filepath, audio_tensor, 24000)
    print(f"Audio guardado en: {output_filepath}")

    print("\n¡Proceso terminado! Revisa el archivo .wav generado en la carpeta de tu proyecto.")
    print(f"Escucha '{output_audio_name}'.")

except Exception as e:
    print(f"\n¡Ocurrió un error! Detalles: {e}")
    print("Posibles causas:")
    print("- Asegúrate de que tienes suficiente espacio en disco para el modelo (varios GB).")
    print("- Asegúrate de que tu conexión a internet esté activa para la descarga automática.")
    "- Si tienes problemas con GPU, verifica tu instalación de CUDA y PyTorch, o prueba en CPU."
    "- Es posible que haya un problema con la instalación de ChatTTS o sus dependencias."
    "- El modelo es pesado. Si se cuelga, reinicia la terminal/VS Code y vuelve a intentarlo."