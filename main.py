import os
import time
import json
import re
import requests
import boto3
import uuid
import asyncio
import ffmpeg
import sys
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from dotenv import load_dotenv

# --- КОНФИГУРАЦИЯ ---
# Загружаем переменные из файла .env
load_dotenv()

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME", "tv-source-files-2026") # Значение по умолчанию
REGION = "ru-central1"

# Проверка наличия ключей при старте
if not YANDEX_API_KEY or not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    print("⚠️  ВНИМАНИЕ: Не найдены ключи доступа в файле .env или переменных окружения.")

# Настройка S3 клиента
session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION
)

app = FastAPI()

# Разрешаем фронтенду общаться с бэкендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Хранилище в памяти (для простоты)
projects_db = {}

# --- ПРОВЕРКИ ПРИ ЗАПУСКЕ ---
@app.on_event("startup")
async def startup_checks():
    print("--- ЗАПУСК ПРОВЕРОК ---")
    
    # 1. Проверка S3
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
        print(f"✅ S3 Бакет '{BUCKET_NAME}' найден и доступен.")
    except Exception as e:
        print(f"❌ ОШИБКА S3: Не удалось получить доступ к бакету '{BUCKET_NAME}'.")
        print(f"   Детали: {e}")
        print("   -> Создайте бакет в консоли Yandex Cloud или проверьте права сервисного аккаунта.")

    # 2. Проверка FFmpeg
    try:
        ffmpeg.input('dummy').output('dummy').run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error:
        # Это нормально, так как input dummy, главное что бинарник запустился
        print("✅ FFmpeg найден.")
    except FileNotFoundError:
        print("❌ ОШИБКА FFmpeg: Программа ffmpeg не найдена в PATH.")
        print("   -> Установите FFmpeg и добавьте его в системные переменные.")
    except Exception:
        # Если ffmpeg установлен, он скорее всего выдаст ошибку на dummy input, но не FileNotFoundError
        print("✅ FFmpeg найден (проверка бинарника).")

# --- ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ ---

class ProjectStatus:
    QUEUED = "В очереди"
    DOWNLOADING = "Скачивание"
    CONVERTING = "Конвертация"
    UPLOADING = "Загрузка в облако"
    TRANSCRIBING = "Распознавание"
    COMPLETED = "Готово"
    ERROR = "Ошибка"

class SpeakerMapping(BaseModel):
    speaker_label: str # "1", "2" (от нейросети)
    mapped_name: str   # "Эфендиева", "АЗК"

class ExportRequest(BaseModel):
    mappings: List[SpeakerMapping]
    filename: str

class CreateProjectRequest(BaseModel):
    url: str

# --- ЛОГИКА ОБРАБОТКИ ---

def parse_filename_metadata(filename):
    """Извлекает имена и таймкоды из названия файла"""
    stop_words = {'лайф', 'лайфы', 'интер', 'синхрон', 'снх', 'бз', 'f8', 'wav', 'mp3', 'mp4', 'mov', 'wmv'}
    result = {"speakers": [], "start_tc": "00:00:00:00"}
    
    # Поиск таймкода
    tc_match = re.search(r'(\d{2}:\d{2}:\d{2}:\d{2})', filename)
    if tc_match:
        result["start_tc"] = tc_match.group(1)
        # Убираем таймкод из имени для чистоты парсинга имен
        filename = filename.replace(result["start_tc"], "")

    clean_name = re.sub(r'\.[^.]+$', '', filename) # убрать расширение
    parts = re.split(r'[,\_]+', clean_name)
    
    for part in parts:
        word = part.strip()
        if word and word.lower() not in stop_words and not re.match(r'\d{2}\.\d{2}\.\d{4}', word):
            result["speakers"].append(word)
    
    return result

def frames_to_tc(frames):
    h = int(frames / (25 * 3600))
    rem = frames % (25 * 3600)
    m = int(rem / (25 * 60))
    rem = rem % (25 * 60)
    s = int(rem / 25)
    f = int(rem % 25)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

def tc_to_frames(tc_str):
    try:
        parts = list(map(int, tc_str.split(':')))
        return (parts[0]*3600 + parts[1]*60 + parts[2])*25 + parts[3]
    except:
        return 0

def process_video_task(project_id: str, disk_url: str):
    local_video_path = f"temp_{project_id}_video"
    local_audio_path = f"temp_{project_id}.opus"
    object_name = f"{project_id}.opus"

    try:
        # 1. СКАЧИВАНИЕ
        projects_db[project_id]["status"] = ProjectStatus.DOWNLOADING
        
        # Получаем прямую ссылку с Яндекс.Диска
        api_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download'
        resp = requests.get(api_url, params={'public_key': disk_url})
        resp.raise_for_status()
        download_url = resp.json()['href']
        
        # Парсим имя файла из API Диска
        original_filename = "video_source.mp4" 
        try:
             meta_url = 'https://cloud-api.yandex.net/v1/disk/public/resources'
             meta_resp = requests.get(meta_url, params={'public_key': disk_url})
             if meta_resp.status_code == 200:
                 original_filename = meta_resp.json()['name']
        except:
            pass
            
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(local_video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # 2. КОНВЕРТАЦИЯ (FFmpeg)
        projects_db[project_id]["status"] = ProjectStatus.CONVERTING
        # Извлекаем аудио в OPUS (требование SpeechKit)
        (
            ffmpeg
            .input(local_video_path)
            .output(local_audio_path, acodec='libopus', ac=1) # Моно
            .overwrite_output()
            .run(quiet=True)
        )

        # 3. ЗАГРУЗКА В S3
        projects_db[project_id]["status"] = ProjectStatus.UPLOADING
        s3.upload_file(local_audio_path, BUCKET_NAME, object_name)
        
        # Генерируем временную подписанную ссылку (Presigned URL)
        # Это позволяет SpeechKit скачать файл даже из приватного бакета
        file_uri = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': object_name},
            ExpiresIn=3600 # Ссылка валидна 1 час
        )

        # 4. РАСПОЗНАВАНИЕ (SpeechKit Long Running)
        projects_db[project_id]["status"] = ProjectStatus.TRANSCRIBING
        
        sk_body = {
            "config": {
                "specification": {
                    "languageCode": "ru-RU",
                    "literature_text": True,
                    "profanityFilter": False,
                    "audioEncoding": "OGG_OPUS"
                }
            },
            "audio": {
                "uri": file_uri
            }
        }
        
        sk_headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}"}
        sk_resp = requests.post(
            "https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize",
            json=sk_body,
            headers=sk_headers
        )
        sk_resp.raise_for_status()
        operation_id = sk_resp.json()['id']

        # Поллинг (ожидание) результата от SpeechKit
        while True:
            time.sleep(3) # Частый поллинг для демо
            op_resp = requests.get(
                f"https://operation.api.cloud.yandex.net/operations/{operation_id}",
                headers=sk_headers
            )
            op_data = op_resp.json()
            if op_data.get('done'):
                break
        
        # Проверка на ошибку внутри операции
        if 'error' in op_data:
             raise Exception(f"SpeechKit Error: {op_data['error']}")

        # 5. ОБРАБОТКА РЕЗУЛЬТАТА
        chunks = op_data['response']['chunks']
        
        # Предварительный маппинг спикеров
        meta = parse_filename_metadata(original_filename)
        projects_db[project_id]["meta"] = meta
        projects_db[project_id]["original_filename"] = original_filename
        
        speaker_durations = {}
        raw_segments = []
        
        start_frames = tc_to_frames(meta['start_tc'])

        for chunk in chunks:
            channel = chunk.get('channelTag', '1')
            if not chunk['alternatives']: continue
            
            text = chunk['alternatives'][0]['text']
            # SpeechKit возвращает время в строках "12.5s"
            start_s_str = chunk['alternatives'][0]['words'][0]['startTime']
            end_s_str = chunk['alternatives'][0]['words'][-1]['endTime']
            
            start_s = float(start_s_str.replace('s',''))
            end_s = float(end_s_str.replace('s',''))
            
            dur = end_s - start_s
            speaker_durations[channel] = speaker_durations.get(channel, 0) + dur
            
            # Рассчитываем абсолютный таймкод
            abs_frames = start_frames + int(start_s * 25)
            tc_formatted = frames_to_tc(abs_frames)
            
            raw_segments.append({
                "timecode": tc_formatted,
                "speaker": channel,
                "text": text
            })

        # Логика назначения имен
        detected_speakers = {}
        sorted_voices = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
        
        file_names = meta['speakers'] # ['Носырев', 'Корр']
        
        for i, (voice_id, dur) in enumerate(sorted_voices):
            suggested = f"Спикер {voice_id}"
            
            # Примитивная логика: пытаемся сопоставить по порядку убывания длительности
            # с именами из файла.
            if i < len(file_names):
                suggested = file_names[i]
            
            detected_speakers[voice_id] = {
                "duration_sec": round(dur, 1),
                "suggested_name": suggested
            }

        projects_db[project_id]["result"] = {
            "segments": raw_segments,
            "speakers": detected_speakers,
            "meta": meta
        }
        projects_db[project_id]["status"] = ProjectStatus.COMPLETED

    except Exception as e:
        print(f"Task Failed: {e}")
        projects_db[project_id]["status"] = ProjectStatus.ERROR
        projects_db[project_id]["error"] = str(e)

    finally:
        # Чистка
        if os.path.exists(local_video_path): os.remove(local_video_path)
        if os.path.exists(local_audio_path): os.remove(local_audio_path)
        # Из бакета удаляем
        try:
            s3.delete_object(Bucket=BUCKET_NAME, Key=object_name)
        except:
            pass

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    """
    Проверка здоровья сервера.
    Если вы видите это, значит бэкенд запущен и работает.
    Перейдите на http://localhost:3000 для работы с интерфейсом.
    """
    return {"status": "ok", "service": "ABTGS Backend", "message": "Go to http://localhost:3000 for UI"}

@app.post("/api/v1/projects")
async def create_project(req: CreateProjectRequest, background_tasks: BackgroundTasks):
    pid = str(uuid.uuid4())
    projects_db[pid] = {
        "id": pid,
        "status": ProjectStatus.QUEUED,
        "created_at": time.time()
    }
    
    background_tasks.add_task(process_video_task, pid, req.url)
    return {"id": pid}

@app.get("/api/v1/projects/{pid}/status")
async def get_status(pid: str):
    if pid not in projects_db:
        raise HTTPException(404, "Project not found")
    proj = projects_db[pid]
    # Возвращаем структуру, которую ждет фронтенд адаптер
    return {
        "status": proj["status"],
        "error": proj.get("error")
    }

@app.get("/api/v1/projects/{pid}")
async def get_result(pid: str):
    proj = projects_db.get(pid)
    if not proj or proj["status"] != ProjectStatus.COMPLETED:
        raise HTTPException(400, "Not ready")
    return proj["result"]

@app.post("/api/v1/projects/{pid}/export")
async def export_docx(pid: str, req: ExportRequest):
    proj = projects_db.get(pid)
    if not proj:
        raise HTTPException(404)
    
    final_map = {m.speaker_label: m.mapped_name for m in req.mappings}
    
    doc = Document()
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    
    filename = proj.get("original_filename", "transcript")
    doc.add_paragraph(f"ИСХОДНИК: {filename}")
    
    segments = proj["result"]["segments"]
    
    for seg in segments:
        speaker_name = final_map.get(seg["speaker"], f"Спикер {seg['speaker']}")
        
        p = doc.add_paragraph()
        p_runner = p.add_run(f"{seg['timecode']} {speaker_name}: ")
        p_runner.bold = True
        p.add_run(seg["text"])
    
    output_filename = f"transcript_{pid}.docx"
    doc.save(output_filename)
    
    from fastapi.responses import FileResponse
    return FileResponse(output_filename, filename=filename.replace(".wmv", ".docx").replace(".mp4", ".docx"))

if __name__ == "__main__":
    import uvicorn
    print("Запуск сервера на http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
