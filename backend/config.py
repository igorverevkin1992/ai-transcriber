import json
import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class JsonFormatter(logging.Formatter):
    """Структурированный JSON-вывод для логов (для парсинга в ELK / Cloud Logging)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()
_handler = logging.StreamHandler()
if LOG_FORMAT == "json":
    _handler.setFormatter(JsonFormatter())
else:
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger("abtgs")

# --- Подавление безвредного «шума» сторонних библиотек ---
# Все эти предупреждения косметические: пайплайн Whisper + диаризация работает
# штатно. Глушим точечно (по тексту/источнику), чтобы НЕ прятать реальные ошибки.

# pyannote жалуется, что torchcodec не загрузился (на Windows не находит FFmpeg-
# DLL). WhisperX отдаёт аудио в pyannote уже декодированным в памяти, поэтому
# torchcodec не нужен — диаризация идёт как обычно.
warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly")
# pyannote отключает TF32 ради воспроизводимости — ожидаемо, на результат не влияет.
warnings.filterwarnings("ignore", message=r"\s*TensorFloat-32 \(TF32\) has been disabled")

# Однострочные INFO/WARNING сторонних библиотек, проходящие через наш хендлер.
_SUPPRESSED_LOG_SUBSTRINGS = (
    "Skipping import of cpp extensions",  # torchao: версия torch несовместима (не используем)
    "Lightning automatically upgraded your loaded checkpoint",  # авто-миграция чекпойнта pyannote
    "Redirects are currently not supported",  # torch на Windows/macOS
)


class _ThirdPartyNoiseFilter(logging.Filter):
    """Глушит известные безвредные строки логов сторонних библиотек."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in _SUPPRESSED_LOG_SUBSTRINGS)


_handler.addFilter(_ThirdPartyNoiseFilter())

# Этот логгер держит собственный хендлер (свой формат «W0622 …»), фильтр выше его
# не ловит — глушим по уровню. (Lightning тоже свой; его уровень поднимается в
# services.py уже ПОСЛЕ импорта whisperx, иначе настройку перетрёт сам Lightning.)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

# --- API Keys ---
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
API_KEY = os.getenv("API_KEY")  # if set, X-API-Key header required on /api/* requests
HF_TOKEN = os.getenv("HF_TOKEN")  # HuggingFace token for pyannote diarization models
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Gemini API for text post-processing
# Строгая диаризация: если пайплайн не может запуститься (нет HF_TOKEN или
# pyannote не загрузился), задача завершается с понятной ошибкой вместо
# молчаливого «весь текст у одного спикера». Отключите (false), чтобы всё же
# получать один-спикерный черновик.
STRICT_DIARIZATION = os.getenv("STRICT_DIARIZATION", "true").lower() in ("1", "true", "yes")

# Модель диаризации pyannote. Новые версии whisperx по умолчанию тянут
# gated-модель `pyannote/speaker-diarization-community-1`, для которой нужно
# отдельно принять условия. По умолчанию закрепляем проверенную `3.1`
# (её условия обычно уже приняты). Переопределите, если приняли community-1.
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

# Точное число говорящих для подсказки pyannote, когда оно известно заранее, а
# имён в имени файла нет (без хинта pyannote на CPU склонна сливать короткие
# реплики ведущего в кластер гостя). Приоритет: токен `sN` в имени файла → этот
# env → число имён из имени файла. 0 = не задано. При >=2 передаётся в диаризацию
# как min_speakers == max_speakers.
DIARIZATION_NUM_SPEAKERS = int(os.getenv("DIARIZATION_NUM_SPEAKERS", "0") or "0")

# Резегментация по словам: whisperx.assign_word_speakers размечает спикера
# КАЖДОМУ слову, но один ASR-сегмент часто захватывает короткую вставку другого
# спикера (вопрос ведущего внутри монолога гостя). Когда true — режем такой
# сегмент на однородные по спикеру куски по word-level меткам, вместо одного
# спикера на весь сегмент. Сегменты без смены спикера внутри не трогаются
# (исходный текст сохраняется дословно). default: true.
WORD_LEVEL_DIARIZATION = os.getenv("WORD_LEVEL_DIARIZATION", "true").lower() in ("1", "true", "yes")

# Резать ASR-сегмент по смене word-метки спикера ТОЛЬКО на конце предложения.
# pyannote нередко ошибочно метит хвост фразы героя соседним спикером посреди
# предложения («…я выиграл тех, кто [там попался]» — «там попался» помечено как
# АЗК). Раньше резегментация отрезала такой хвост и приклеивала к началу реплики
# ведущего. Когда true, смена спикера признаётся границей реплики лишь если
# предыдущее слово закончило предложение (.!?…); флип посреди фразы считается
# шумом разметки, и слово остаётся в текущей реплике. Настоящие вставки ведущего
# («…культивируем. Вы со студентами часто играете?») режутся как прежде, т.к.
# стоят после конца предложения. false — резать на любой смене (прежнее
# поведение). default: true. Действует только при WORD_LEVEL_DIARIZATION=true.
WORD_SPLIT_SENTENCE_BOUNDARY = os.getenv("WORD_SPLIT_SENTENCE_BOUNDARY", "true").lower() in ("1", "true", "yes")

# Свернуть реплики неназванного «лишнего» спикера в «(Технические моменты)».
# Диаризация нередко выделяет члена съёмочной группы в отдельный кластер
# («Спикер N»), который не является ни интервьюером, ни названным гостем (имя
# не задано в файле и не выводится из обращений в диалоге). Такие реплики —
# техническая болтовня группы (команды оператору, «в моторе», «допишем»), и в
# эталонах они помечены маркером техмомента, а не показаны спикером. Когда true,
# реплики каждого генерик-спикера «Спикер N» с долей речи <= UNNAMED_SPEAKER_TECH_MAX_RATIO
# заменяются на «(Технические моменты)», а сам спикер убирается из легенды.
# Доля-гард защищает реального, но неузнанного по имени гостя (он обычно говорит
# заметно дольше). default: false (консервативно — риск задеть редкого гостя).
UNNAMED_SPEAKER_AS_TECH = os.getenv("UNNAMED_SPEAKER_AS_TECH", "false").lower() in ("1", "true", "yes")
UNNAMED_SPEAKER_TECH_MAX_RATIO = float(os.getenv("UNNAMED_SPEAKER_TECH_MAX_RATIO", "0.15") or "0.15")

# --- Paths ---
TEMP_DIR = Path("temp_files")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("completed_docx")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Limits ---
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB

# --- Turn building (склейка реплик в абзацы, как в эталонных стенограммах) ---
TURN_MERGE_ENABLED = os.getenv("TURN_MERGE_ENABLED", "true").lower() in ("1", "true", "yes")
TURN_INLINE_TC_SECONDS = float(os.getenv("TURN_INLINE_TC_SECONDS", "60"))
TECH_BREAK_GAP_SECONDS = float(os.getenv("TECH_BREAK_GAP_SECONDS", "30"))
# Ставить ли точку после маркера техпаузы: «(Технические моменты).» (true, как в
# эталонах f7/f8) или «(Технические моменты)» без точки (как в некоторых других).
TECH_BREAK_DOT = os.getenv("TECH_BREAK_DOT", "true").lower() in ("1", "true", "yes")

# --- Whisper tuning ---
# Порог ЛИДИРУЮЩЕГО маркера техмоментов: маркер в начале документа, если первая
# реплика начинается ≥ этого числа секунд от старта записи. Меньше общего
# TECH_BREAK_GAP_SECONDS (30): эталоны отмечают и короткий стартовый футаж
# (ф4: вход в комнату, 25 c). Откат к прежнему поведению: env=30.
TECH_BREAK_LEAD_SECONDS = float(os.getenv("TECH_BREAK_LEAD_SECONDS", "15") or "15")

# beam_size: выше = точнее, но медленнее (рекомендуется 5-10)
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
# Подсказка декодеру: контекст и стиль; имена спикеров из имени файла
# добавляются автоматически (см. services._build_whisper_prompt)
WHISPER_INITIAL_PROMPT = os.getenv("WHISPER_INITIAL_PROMPT", "Интервью на русском языке.")
# Глоссарий правильных написаний (имена, термины, названия), через запятую.
# Подмешивается в initial_prompt Whisper и в промпт Gemini, чтобы снизить
# ошибки в редких именах собственных и иностранных терминах.
# Пример: "Мордюкова, Гурченко, Мосфильм, star quality, НТВ"
TRANSCRIPT_GLOSSARY = os.getenv("TRANSCRIPT_GLOSSARY", "").strip()

# Детерминированные замены "неверно=>верно" (пары через запятую, разделитель
# внутри пары — "=>"). Применяются ВСЕГДА в regex_cleanup, независимо от Gemini,
# поэтому гарантированно чинят известные ошибки распознавания редких имён/терминов.
# Пример: "Мурдюкова=>Мордюкова,Горченко=>Гурченко,прияды=>плеяда,просто квашено=>Простоквашино"
GLOSSARY_REPLACEMENTS = os.getenv("GLOSSARY_REPLACEMENTS", "").strip()

# --- Автоопределение интервьюера (АЗК) ---
# Интервьюер за кадром чередуется со всеми гостями; помечаем его меткой АЗК и
# исключаем из легенды (см. docx_export.is_legend_excluded_name).
INTERVIEWER_AUTODETECT = os.getenv("INTERVIEWER_AUTODETECT", "true").lower() in ("1", "true", "yes")
INTERVIEWER_LABEL = os.getenv("INTERVIEWER_LABEL", "АЗК")
# Именовать интервьюера по прямому обращению из диалога («Яна, …» → «Яна»
# вместо АЗК): ф14-эталон именует ведущую в кадре по имени; АЗК остаётся для
# безымянных закадровых (ф4: имя ведущей в аудио не звучит — там АЗК).
INTERVIEWER_NAME_FROM_DIALOGUE = os.getenv("INTERVIEWER_NAME_FROM_DIALOGUE", "true").lower() in ("1", "true", "yes")
# В интервью «один на один» (2 спикера) чередование ролей не различает, поэтому
# интервьюер помечается только когда сходятся ДВА независимых сигнала: он
# заметно чаще задаёт вопросы И говорит меньше гостя (см. detect_interviewer).
# Дефолт true с ф4: без пометки имя из файла уходило ведущей, а гость становился
# «Спикер 2». При противоречии сигналов пометки нет. Откат: env=false.
INTERVIEWER_LABEL_SINGLE_GUEST = os.getenv("INTERVIEWER_LABEL_SINGLE_GUEST", "true").lower() in ("1", "true", "yes")
INTERVIEWER_MIN_DISTINCT_GUESTS = int(os.getenv("INTERVIEWER_MIN_DISTINCT_GUESTS", "2"))
INTERVIEWER_MAJORITY_RATIO = float(os.getenv("INTERVIEWER_MAJORITY_RATIO", "0.5"))

# --- Авто-определение имён гостей из диалога ---
# Если в имени файла нет имён, попытаться определить имя-отчество гостя по тому,
# как к нему обращаются в репликах («Олег Александрович, …»). При наличии
# GEMINI_API_KEY использует Gemini, иначе — детерминированную эвристику по
# вокативам. Интервьюер остаётся АЗК (его имя в аудио не звучит). default: true.
SPEAKER_NAME_AUTODETECT = os.getenv("SPEAKER_NAME_AUTODETECT", "true").lower() in ("1", "true", "yes")

# --- ASR anti-hallucination ---
# Известные Whisper-галлюцинации на музыке/тишине (подстрочный поиск,
# регистронезависимо, только для коротких сегментов). Env заменяет список.
_DEFAULT_HALLUCINATIONS = (
    "субтитры,редактор субтитров,продолжение следует,спасибо за просмотр,"
    "dimatorzok,дима торжок,подпишись,подписывайтесь на канал"
)
HALLUCINATION_BLACKLIST = [
    p.strip().lower()
    for p in os.getenv("HALLUCINATION_BLACKLIST", _DEFAULT_HALLUCINATIONS).split(",")
    if p.strip()
]
# Сегмент с avg_logprob ниже порога — неразборчивая речь: человек пишет
# "(неразборчиво)" (цензус: 8 случаев в эталонах), а не выдуманный текст.
UNCLEAR_LOGPROB_THRESHOLD = float(os.getenv("UNCLEAR_LOGPROB_THRESHOLD", "-1.2"))
# Сегмент с no_speech_prob выше порога — не речь, отбрасывается.
NO_SPEECH_PROB_THRESHOLD = float(os.getenv("NO_SPEECH_PROB_THRESHOLD", "0.85"))

# --- SpeechKit tuning ---
SPEECHKIT_LITERATURE_TEXT = os.getenv("SPEECHKIT_LITERATURE_TEXT", "false").lower() in ("1", "true", "yes")

# --- Gemini tuning ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Более сильная модель для «умных» пассов (правка границ спикеров, техмоменты,
# имена, паспорт): рассуждение о смысле диалога, где flash слабее. Полировка
# текста остаётся на дешёвой GEMINI_MODEL. Пусто → используется GEMINI_MODEL.
GEMINI_MODEL_SMART = os.getenv("GEMINI_MODEL_SMART", "").strip() or GEMINI_MODEL
# Проверка доступности Gemini при старте сервера (один тестовый вызов): сбой
# ключа/прокси виден сразу в логе, а не как молчаливая деградация пассов.
GEMINI_HEALTHCHECK = os.getenv("GEMINI_HEALTHCHECK", "true").lower() in ("1", "true", "yes")
# Таймаут одного HTTP-запроса к Gemini, секунды. Без него зависший сокет
# (прокси-«чёрная дыра») держал вызов неограниченно: ретрай-петля _gemini_call
# ловит только ошибки, а не молчание. Покрывает и vision-вызовы (общий клиент).
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "120"))
# Консервативное определение «технических моментов» (реплики съёмочной группы,
# перезапуски камеры, проверка микрофона и т.п.) через Gemini: ЯВНО не-интервью
# фрагменты заменяются маркером «(Технические моменты).». Требует GEMINI_API_KEY.
TECH_MOMENT_DETECTION = os.getenv("TECH_MOMENT_DETECTION", "true").lower() in ("1", "true", "yes")
# Агрессивный режим определения техмоментов: помимо команд группе/настройки
# техники также помечает закадровую организационную болтовню (обращения к
# посторонним по имени, «пойду умоюсь», «давай с тебя»), репетиции и
# обсуждение дублей. Дефолт true с ф4: человеческие эталоны стабильно
# сворачивают репетиционно-организационную часть в маркеры; консервативный
# режим оставлял её репликами. Откат: TECH_MOMENT_AGGRESSIVE=false.
TECH_MOMENT_AGGRESSIVE = os.getenv("TECH_MOMENT_AGGRESSIVE", "true").lower() in ("1", "true", "yes")

# Gemini-правка границ спикеров: pyannote изредка относит короткую реплику ЦЕЛИКОМ
# не тому спикеру (ответ гостя в абзаце ведущего и наоборот). Отдельный проход
# читает диалог с текущими метками + контекст (ведущий АЗК закадровый, гости =
# имена героев, тема из паспорта) и переназначает ТОЛЬКО высокоуверенно
# перепутанные реплики. Дефолт true с ф4 (эталон разделяет склейки
# «вопрос—“Да.”—вопрос», которые диаризация слила в один абзац; без
# GEMINI_API_KEY пасс мягко деградирует в no-op). Откат: env=false.
SPEAKER_BOUNDARY_CORRECTION = os.getenv("SPEAKER_BOUNDARY_CORRECTION", "true").lower() in ("1", "true", "yes")

# --- Видео-описания техмоментов (Gemini Vision) ---
# Обогащать маркеры «(Технические моменты)» описанием происходящего в кадре:
# «(Технические моменты. Съемка нарезки помидоров двумя поварами)» — как в
# человеческих эталонах. Извлекает 1-2 кадра на маркер (ffmpeg) и описывает их
# мультимодальным Gemini. default: false (opt-in: время/стоимость; требует
# GEMINI_API_KEY, ffmpeg и исходное видео). Логика в backend/tech_vision.py.
TECH_MOMENT_VISION = os.getenv("TECH_MOMENT_VISION", "false").lower() in ("1", "true", "yes")
# Максимум маркеров с описаниями на файл (остальные остаются «голыми»).
TECH_VISION_MAX_MARKERS = int(os.getenv("TECH_VISION_MAX_MARKERS", "30") or "30")
# Маркеры короче этого интервала (сек) не описываются — кадр малоинформативен.
TECH_VISION_MIN_GAP_SECONDS = float(os.getenv("TECH_VISION_MIN_GAP_SECONDS", "3") or "3")
# Разрезать длинные интервалы техмоментов по сменам плана в видеоряде (ffmpeg
# scene detection): эталоны ставят маркер на каждый дубль, а аудио границ дублей
# не слышит. Работает только при TECH_MOMENT_VISION=true (без описаний
# одинаковые маркеры подряд бессмысленны). THRESHOLD — чувствительность смены
# плана (0-1); MIN_SPAN — резать только интервалы длиннее этого (сек).
TECH_MARKER_SCENE_SPLIT = os.getenv("TECH_MARKER_SCENE_SPLIT", "true").lower() in ("1", "true", "yes")
# 0.22: склейки между дублями в статичной студийной сцене (тот же план, та же
# комната) дают scene-score мягче межсценовых переходов — 0.30 их пропускал
# (ф4: блок дублей Happy birthday 6→1). Ложный лишний маркер безвреден.
TECH_SCENE_THRESHOLD = float(os.getenv("TECH_SCENE_THRESHOLD", "0.22") or "0.22")
TECH_SCENE_MIN_SPAN = float(os.getenv("TECH_SCENE_MIN_SPAN", "25") or "25")

# --- OCR выжженного в кадр таймкода ---
# Когда стартовый ТК не задан ни в имени файла, ни в метаданных контейнера —
# попытаться распознать выжженный в кадр SMPTE-таймкод (easyocr). default: true.
OCR_TIMECODE = os.getenv("OCR_TIMECODE", "true").lower() in ("1", "true", "yes")
# Область кадра с таймкодом, доли 0-1: "left,top,right,bottom". По умолчанию —
# ВСЯ нижняя треть кадра: бокс ТК встречается и по центру снизу (ф13), и справа;
# прежний дефолт 0.3,0.7 обрезал левый разряд центрального бокса. Пустая строка →
# искать по всему кадру.
OCR_TIMECODE_REGION = os.getenv("OCR_TIMECODE_REGION", "0.0,0.65,1.0,1.0")
# Кусочная коррекция таймкодов по OCR-якорям. Free-run ТК плёнки бежит во время
# пауз записи между дублями, из-за чего «стартовый ТК + медиа-время» дрейфует
# (ф4: +15 c к интервью). Якоря — чтения выжженного ТК каждые ~ANCHOR_INTERVAL
# секунд по всему файлу; сегменты корректируются от ближайшего предыдущего
# якоря. Мягкая деградация: без easyocr/при сбое остаётся линейная модель.
OCR_TC_ANCHORS = os.getenv("OCR_TC_ANCHORS", "true").lower() in ("1", "true", "yes")
OCR_TC_ANCHOR_INTERVAL = float(os.getenv("OCR_TC_ANCHOR_INTERVAL", "120") or "120")

# --- DOCX formatting ---
# Тире между именем и аббревиатурой в легенде: «Имя – ОА» (en-dash, как f7/f8)
# или «Имя — ОА» (em-dash, как в некоторых эталонах). Задайте LEGEND_DASH=—.
LEGEND_DASH = os.getenv("LEGEND_DASH", "–")
# Точка в конце строки легенды/разделителя: «Имя – ОА.» (true, как f7/f8) или
# «Имя — ОА» без точки (false).
LEGEND_TRAILING_DOT = os.getenv("LEGEND_TRAILING_DOT", "true").lower() in ("1", "true", "yes")
# Легенда «списком» (ф14): строки завершаются запятой, последняя — точкой
# («БАТЫРШИНА ЯНА – БЯ,» / «КАНАЕВА СВЕТЛАНА – КС.»). false — прежний стиль.
LEGEND_LIST_STYLE = os.getenv("LEGEND_LIST_STYLE", "false").lower() in ("1", "true", "yes")
# Вставлять ли внутренние разделители-легенды перед первой репликой каждого
# гостя (при >1 госте). true = текущее поведение; false — единый блок легенды
# сверху, без повторов (как в бадминтонном эталоне).
SECTION_DIVIDERS_ENABLED = os.getenv("SECTION_DIVIDERS_ENABLED", "true").lower() in ("1", "true", "yes")
# Пустой абзац между подряд идущими маркерами техмоментов: эталон визуально
# делит репетиционную часть на блоки-дубли (ф4).
TECH_BLOCK_EMPTY_LINE = os.getenv("TECH_BLOCK_EMPTY_LINE", "true").lower() in ("1", "true", "yes")
# Титульная строка документа (имя файла ПРОПИСНЫМИ) перед легендой. Конвенция
# f7/f8 (default true); эталон ф13 начинается сразу с легенды → false.
DOC_TITLE_ENABLED = os.getenv("DOC_TITLE_ENABLED", "true").lower() in ("1", "true", "yes")
# Двухбуквенные аббревиатуры для двухсловных имён БЕЗ отчества: «Арнальди
# Федерико» → «АФ», «Морфео Доменико» → «МД» (как в эталоне ф13). Дефолт false —
# одна буква («Майданов Денис» → «М», как в эталонах f7/f8). Имя-отчество всегда
# даёт инициалы (ОА/ГВ) независимо от флага.
ABBR_TWO_LETTER = os.getenv("ABBR_TWO_LETTER", "false").lower() in ("1", "true", "yes")

# --- DOCX metadata ---
DOCX_AUTHOR = os.getenv("DOCX_AUTHOR", "")
# Шаблон-пакет из эталонной стенограммы (styles/theme/settings/header).
# Генератор открывает его вместо пустого Document(), чтобы не нести отпечатки
# дефолтного шаблона python-docx. Изготавливается scripts/make_template.py.
DOCX_TEMPLATE_PATH = os.getenv(
    "DOCX_TEMPLATE_PATH",
    str(Path(__file__).resolve().parent / "transcript_template.docx"),
)


def _auto_detect_concurrent_tasks() -> int:
    """Автоопределение MAX_CONCURRENT_TASKS на основе VRAM (если есть GPU)."""
    env_val = os.getenv("MAX_CONCURRENT_TASKS")
    if env_val:
        return int(env_val)
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # WhisperX large + pyannote ~8GB per task
            return max(1, int(vram_gb / 8))
    except ImportError:
        pass
    return 2


MAX_CONCURRENT_TASKS = _auto_detect_concurrent_tasks()
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".mov", ".mxf", ".mp4", ".wmv", ".avi", ".mkv", ".ogg", ".flac"}
ALLOWED_URL_HOSTS = {"yadi.sk", "disk.yandex.ru", "disk.yandex.com"}

# При старте автоматически возобновлять прерванные задачи (true) или только
# помечать их как прерванные, оставляя ручной запуск через кнопку «Продолжить»
# (false). Дефолт true: с чекпоинтом распознавания повтор после падения в
# постобработке стоит минуты, а после обрыва транскрипции продолжение всё равно
# требует полного прогона — автоматика избавляет от ручного клика.
AUTO_RECOVER_ON_STARTUP = os.getenv("AUTO_RECOVER_ON_STARTUP", "true").lower() in ("1", "true", "yes")

# --- CORS ---
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "projects.db")

# --- CORS ---
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
