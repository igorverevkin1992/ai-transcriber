"""Prometheus метрики для мониторинга обработки.

Все метрики опциональны: если prometheus_client не установлен, no-op.
Endpoint /metrics включается автоматически в main.py.
"""

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"

    class _Noop:
        def __init__(self, *a, **kw): pass
        def labels(self, *a, **kw): return self
        def inc(self, *a, **kw): pass
        def dec(self, *a, **kw): pass
        def set(self, *a, **kw): pass
        def observe(self, *a, **kw): pass
        def time(self):
            class _Timer:
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _Timer()

    Counter = Gauge = Histogram = _Noop

    def generate_latest() -> bytes:
        return b"# prometheus_client not installed\n"


transcribe_duration = Histogram(
    "abtgs_transcribe_seconds",
    "Время транскрипции в секундах",
    labelnames=("engine",),
    buckets=(10, 30, 60, 180, 300, 600, 1800, 3600, 7200),
)

projects_total = Counter(
    "abtgs_projects_total",
    "Общее число созданных проектов",
    labelnames=("engine",),
)

project_errors = Counter(
    "abtgs_project_errors_total",
    "Ошибки обработки проектов",
    labelnames=("stage",),
)

active_projects = Gauge(
    "abtgs_active_projects",
    "Текущее число обрабатываемых проектов",
)

gemini_calls = Counter(
    "abtgs_gemini_calls_total",
    "Вызовы Gemini API",
    labelnames=("outcome",),
)

diarization_speakers = Histogram(
    "abtgs_diarization_speakers",
    "Число обнаруженных спикеров",
    buckets=(1, 2, 3, 4, 5, 6, 8, 10, 20),
)
