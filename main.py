import os
import torch
import soundfile as sf
import whisper
import re
import ChatTTS
import numpy as np
import moviepy.editor as mp

# --- GLOBAL CONFIGURATION ---
# Folder paths
project_root = os.path.dirname(os.path.abspath(__file__))

AUDIO_DIR = os.path.join(project_root, "audio")
SUBTITLES_DIR = os.path.join(project_root, "subtitles")
VIDEOS_DIR = os.path.join(project_root, "videos")
FINAL_VIDEO_DIR = os.path.join(project_root, "final_video")
TEXT_DIR = os.path.join(project_root, "texts")
VOICES_DIR = os.path.join(project_root, "voices")

# File names
input_text_filename = "input_text.txt"
input_video_filename = "Minecraft.mp4"
output_audio_filename = "output_generated_audio.wav"
output_srt_filename = "output_generated_subtitles.srt"
output_video_filename = "output_video_with_subtitles.mp4"
speaker_voice_filename = "speaker_women.pt"

# Full path construction
input_text_path = os.path.join(TEXT_DIR, input_text_filename)
input_video_path = os.path.join(VIDEOS_DIR, input_video_filename)
output_audio_path = os.path.join(AUDIO_DIR, output_audio_filename)
output_srt_path = os.path.join(SUBTITLES_DIR, output_srt_filename)
output_video_path = os.path.join(FINAL_VIDEO_DIR, output_video_filename)
speaker_voice_path = os.path.join(VOICES_DIR, speaker_voice_filename)

# Video dimensions for TikTok (vertical)
final_video_size = (1080, 1920)

# Device configuration (CPU/GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando el dispositivo: {device}")

# --- MODEL INITIALIZATION ---
try:
    print("\nInicializando ChatTTS...")
    chat = ChatTTS.Chat()
    chat.load(compile=False, device=device)
    print("¡Modelos de ChatTTS cargados correctamente!")
except Exception as e:
    print(f"Error al cargar ChatTTS: {e}")
    exit()

# --- HELPER FUNCTIONS ---
def parse_time(time_str):
    """Convierte un string de tiempo (HH:MM:SS,mmm) a segundos."""
    parts = re.split('[:,]', time_str)
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    milliseconds = int(parts[3])
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

def format_timestamp(seconds, always_include_hours=False, decimal_marker=','):
    """Formatea segundos en una cadena de tiempo SRT."""
    if seconds is None: return "00:00:00,000"

    milliseconds = int(round(seconds * 1000.0))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    if always_include_hours or hours > 0:
        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    else:
        timestamp = f"{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    return timestamp

def srt_to_subtitles_clips(segments, video_size):
    """
    Procesa segmentos de transcripción (con timestamps de palabras) y crea una lista de TextClip para cada subtítulo,
    aplicando estilo personalizado y limitando a un máximo de 3 palabras por subtítulo,
    mostrando todo el texto en segmentos.
    """
    subtitles = []

    for segment in segments:
        # Usar los timestamps a nivel de palabra si están disponibles
        if 'words' in segment and segment['words']:
            words_data = segment['words']
            num_words = len(words_data)

            current_word_index = 0
            while current_word_index < num_words:
                # Obtiene las palabras para el fragmento de subtítulo actual (máximo 3)
                chunk_words_data = words_data[current_word_index : current_word_index + 3]
                
                # Obtiene el texto a mostrar, eliminando espacios extra
                display_text = " ".join([word_info['word'].strip() for word_info in chunk_words_data])

                # Obtiene el tiempo de inicio y fin preciso para este fragmento
                # El inicio es el inicio de la primera palabra del chunk
                chunk_start_time = chunk_words_data[0]['start']
                # El fin es el fin de la última palabra del chunk
                chunk_end_time = chunk_words_data[-1]['end']

                # Crea el TextClip para este fragmento
                try:
                    txt_clip = mp.TextClip(
                        display_text,
                        fontsize=120,          # Tamaño de fuente aumentado para mejor visibilidad
                        color='white',         # Color de relleno blanco
                        font='Impact',         # Intentando la fuente 'Impact' para un aspecto audaz
                        stroke_color='black',  # Color de trazo negro
                        stroke_width=6,        # Ancho de trazo aumentado para un contorno más prominente
                        size=(video_size[0] * 0.9, None), # Cuadro de texto más ancho para acomodar una fuente más grande
                        method='caption'       # Todavía útil para el ajuste interno si una palabra es demasiado larga
                    )
                except Exception as e:
                    print(f"Error al cargar la fuente 'Impact': {e}. Intentando 'Arial-Bold'.")
                    try:
                        txt_clip = mp.TextClip(
                            display_text,
                            fontsize=120,
                            color='white',
                            font='Arial-Bold', # Fallback a Arial-Bold
                            stroke_color='black',
                            stroke_width=6,
                            size=(video_size[0] * 0.9, None),
                            method='caption'
                        )
                    except Exception as e_fallback:
                        print(f"Error al cargar la fuente 'Arial-Bold': {e_fallback}. Volviendo a la fuente por defecto.")
                        txt_clip = mp.TextClip(
                            display_text,
                            fontsize=120,
                            color='white',
                            stroke_color='black',
                            stroke_width=6,
                            size=(video_size[0] * 0.9, None),
                            method='caption'
                        )

                txt_clip = txt_clip.set_start(chunk_start_time).set_duration(chunk_end_time - chunk_start_time)
                txt_clip = txt_clip.set_position('center') # Mantener el texto centrado
                subtitles.append(txt_clip)

                current_word_index += 3 # Avanza al siguiente fragmento de 3 palabras
        else:
            # Fallback si por alguna razón no hay timestamps de palabras (menos preciso)
            print(f"Advertencia: No se encontraron timestamps de palabras para el segmento: {segment.get('text', 'N/A')}. Usando tiempos de segmento.")
            display_text = segment['text'].strip()
            start_time = segment['start']
            end_time = segment['end']

            words = display_text.split()
            num_words = len(words)
            segment_duration = end_time - start_time
            
            if num_words > 0:
                duration_per_word = segment_duration / num_words
            else:
                duration_per_word = 0

            current_word_index_fallback = 0
            while current_word_index_fallback < num_words:
                chunk_words = words[current_word_index_fallback : current_word_index_fallback + 3]
                display_text_fallback = " ".join(chunk_words)

                chunk_start_time_fallback = start_time + (current_word_index_fallback * duration_per_word)
                chunk_end_time_fallback = chunk_start_time_fallback + (len(chunk_words) * duration_per_word)
                
                if chunk_end_time_fallback > end_time:
                    chunk_end_time_fallback = end_time

                try:
                    txt_clip = mp.TextClip(
                        display_text_fallback,
                        fontsize=120,
                        color='white',
                        font='Impact',
                        stroke_color='black',
                        stroke_width=6,
                        size=(video_size[0] * 0.9, None),
                        method='caption'
                    )
                except Exception as e:
                    print(f"Error al cargar la fuente 'Impact': {e}. Intentando 'Arial-Bold'.")
                    try:
                        txt_clip = mp.TextClip(
                            display_text_fallback,
                            fontsize=120,
                            color='white',
                            font='Arial-Bold',
                            stroke_color='black',
                            stroke_width=6,
                            size=(video_size[0] * 0.9, None),
                            method='caption'
                        )
                    except Exception as e_fallback:
                        print(f"Error al cargar la fuente 'Arial-Bold': {e_fallback}. Volviendo a la fuente por defecto.")
                        txt_clip = mp.TextClip(
                            display_text_fallback,
                            fontsize=120,
                            color='white',
                            stroke_color='black',
                            stroke_width=6,
                            size=(video_size[0] * 0.9, None),
                            method='caption'
                        )

                txt_clip = txt_clip.set_start(chunk_start_time_fallback).set_duration(chunk_end_time_fallback - chunk_start_time_fallback)
                txt_clip = txt_clip.set_position('center')
                subtitles.append(txt_clip)
                current_word_index_fallback += 3

    return subtitles

# --- FUNCIONES PRINCIPALES ---
def generate_chattts_audio():
    """Lee el texto de un archivo y lo convierte a voz."""
    try:
        with open(input_text_path, 'r', encoding='utf-8') as f:
            text_to_convert = f.read()
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{input_text_path}'. Creando uno de ejemplo.")
        os.makedirs(TEXT_DIR, exist_ok=True)
        with open(input_text_path, 'w', encoding='utf-8') as f:
            f.write("Hola, este es un texto de prueba para la conversión de voz.")
        text_to_convert = "Hola, este es un texto de prueba para la conversión de voz."

    print(f"\nGenerando audio para el texto con {speaker_voice_path}:")
    print(text_to_convert)

    try:
        spk = torch.load(speaker_voice_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del hablante en '{speaker_voice_path}'.")
        return None

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk,
        temperature=0.3,
        top_P=0.7,
        top_K=20,
    )
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_2][laugh_0][break_6]',
    )

    wavs = chat.infer(
        [text_to_convert],
        params_refine_text=params_refine_text,
        params_infer_code=params_infer_code,
    )

    if isinstance(wavs[0], torch.Tensor):
        audio_data = wavs[0].cpu().numpy()
    elif isinstance(wavs[0], np.ndarray):
        audio_data = wavs[0]
    else:
        raise TypeError(f"El tipo de dato de audio retornado no es soportado: {type(wavs[0])}")

    os.makedirs(AUDIO_DIR, exist_ok=True)
    sf.write(output_audio_path, audio_data, 24000)
    print(f"Audio guardado en: {output_audio_path}")
    return output_audio_path

def transcribe_audio_to_srt(audio_file_path):
    """Transcribe un archivo de audio y guarda el resultado como subtítulos SRT."""
    print(f"\nIniciando transcripción de audio: {audio_file_path}")

    try:
        # Habilitar word_timestamps para obtener tiempos precisos por palabra
        model = whisper.load_model("small", device=device)
        result = model.transcribe(audio_file_path, fp16=False, word_timestamps=True)

        os.makedirs(SUBTITLES_DIR, exist_ok=True)
        with open(output_srt_path, "w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(result["segments"]):
                # El archivo SRT sigue conteniendo el segmento completo,
                # pero los clips de video usarán los tiempos de palabra para mayor precisión.
                start_time = format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')
                end_time = format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')
                srt_file.write(f"{i + 1}\n")
                srt_file.write(f"{start_time} --> {end_time}\n")
                srt_file.write(f"{segment['text'].strip()}\n\n")

        print(f"Transcripción y subtítulos guardados en: {output_srt_path}")
        return result # Retornar el objeto completo 'result' que contiene 'segments' y 'words'

    except Exception as e:
        print(f"Error durante la transcripción: {e}")
        return None

def create_final_video(srt_data): # Ahora recibe los datos de transcripción
    """
    Combina el video, el audio generado y los subtítulos en un video final.
    Ahora el video se recorta, redimensiona y ajusta su duración.
    """
    try:
        print(f"\nCargando video: {input_video_path}")
        video_clip = mp.VideoFileClip(input_video_path)

        w, h = video_clip.size
        target_w, target_h = final_video_size

        if w > h:
            crop_width = h * target_w / target_h
            x1 = (w - crop_width) / 2
            x2 = x1 + crop_width
            video_clip = video_clip.crop(x1=x1, width=crop_width)

        final_clip = video_clip.resize(newsize=final_video_size)

        print(f"Cargando audio: {output_audio_path}")
        audio_clip = mp.AudioFileClip(output_audio_path)

        # Establece la duración del video final para que coincida con la duración del audio
        final_clip = final_clip.set_duration(audio_clip.duration)

        # El audio del video original es reemplazado por el nuevo audio
        final_clip = final_clip.set_audio(audio_clip)

        print("Generando subtítulos desde los datos de transcripción...")
        # Pasar directamente los segmentos de la transcripción a la función de clips
        subtitles_clips = srt_to_subtitles_clips(srt_data["segments"], final_clip.size)

        final_video = mp.CompositeVideoClip([final_clip] + subtitles_clips)

        os.makedirs(FINAL_VIDEO_DIR, exist_ok=True)
        print(f"Escribiendo el archivo de salida: {output_video_path}")
        final_video.write_videofile(
            output_video_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=final_video.fps
        )

        final_video.close()
        video_clip.close()
        audio_clip.close()
        print("\n¡Proceso de creación de video finalizado con éxito!")
    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar el archivo necesario: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado al crear el video: {e}")

if __name__ == "__main__":
    audio_file = generate_chattts_audio()
    if audio_file:
        srt_result = transcribe_audio_to_srt(audio_file)
        if srt_result: # Asegurarse de que la transcripción fue exitosa
            create_final_video(srt_result)
