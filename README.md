# ABTGS — Automated Broadcast Transcript Generation System

Система автоматической генерации расшифровок эфирных материалов для телевизионного производства. Обрабатывает видео/аудиофайлы, выполняет распознавание речи с диаризацией дикторов и формирует готовые документы Word с покадровыми таймкодами (SMPTE 25 FPS).

## Возможности

- Автоматическое распознавание речи через Yandex SpeechKit
- Диаризация — определение и разделение дикторов
- Покадровые таймкоды в формате SMPTE (HH:MM:SS:FF, 25 кадров/сек)
- Интерактивная верификация дикторов в веб-интерфейсе
- Экспорт в DOCX с форматированной расшифровкой
- Загрузка материалов по ссылке Яндекс.Диска
- Поддержка множества видео- и аудиоформатов

## Стек технологий

| Слой | Технологии |
|------|-----------|
| Frontend | React 18, TypeScript, Vite 5, Tailwind CSS, lucide-react |
| Backend | Python, FastAPI, Uvicorn |
| Распознавание речи | Yandex SpeechKit API |
| Облачное хранилище | Yandex Cloud Object Storage (S3-совместимое) |
| Медиа-обработка | FFmpeg (libopus) |
| Генерация документов | python-docx |

## Структура проекта

```
ai-transcriber/
├── main.py                     # FastAPI-бэкенд (API, обработка, генерация DOCX)
├── App.tsx                     # Корневой React-компонент (состояние, маршрутизация)
├── index.html                  # HTML-точка входа
├── index.tsx                   # Монтирование React DOM
├── types.ts                    # TypeScript-интерфейсы
├── config.ts                   # Конфигурация API URL
├── components/
│   ├── UploadForm.tsx          # Форма загрузки (ссылка Яндекс.Диска)
│   ├── ProcessingStatus.tsx    # Индикатор прогресса обработки
│   ├── VerificationDashboard.tsx # Панель верификации дикторов
│   ├── SpeakerMatrix.tsx       # Матрица идентификации дикторов
│   └── TranscriptPreview.tsx   # Предпросмотр расшифровки
├── services/
│   ├── api.ts                  # API-клиент для взаимодействия с бэкендом
│   └── mockData.ts             # Тестовые данные
├── package.json                # Node.js-зависимости
├── requirements.txt            # Python-зависимости
├── vite.config.ts              # Конфигурация Vite
├── tsconfig.json               # Конфигурация TypeScript
├── start.sh                    # Скрипт запуска обоих серверов
├── .env.example                # Шаблон переменных окружения
└── .env.local                  # Локальная конфигурация
```

## Требования

- Python 3.8+
- Node.js 16+
- FFmpeg (установлен и доступен в PATH)
- Ключ API Yandex SpeechKit
- Учётные данные Yandex Cloud Object Storage

## Установка и запуск

### 1. Клонирование и настройка окружения

```bash
git clone <url-репозитория>
cd ai-transcriber
cp .env.example .env
```

Заполните `.env` своими ключами:

```env
YANDEX_API_KEY=ваш_ключ_yandex_speechkit
AWS_ACCESS_KEY_ID=ваш_access_key_id
AWS_SECRET_ACCESS_KEY=ваш_secret_access_key
BUCKET_NAME=tv-source-files-2026
```

### 2. Backend

```bash
pip install -r requirements.txt
python main.py
```

Сервер запустится на `http://localhost:8000`.

### 3. Frontend

В отдельном терминале:

```bash
npm install
npm run dev
```

Интерфейс будет доступен на `http://localhost:3000`.

### Быстрый запуск (оба сервера)

```bash
./start.sh
```

## Запуск в облаке (Project IDX)

1. Дождитесь **Rebuild Environment** (установит ffmpeg и python).
2. Откройте терминал и запустите бэкенд:
   ```bash
   pip install -r requirements.txt
   python main.py
   ```
3. Во **втором** терминале запустите фронтенд:
   ```bash
   npm install
   npm run dev
   ```
4. Нажмите **Project Previews** для открытия приложения.

## API

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| GET | `/` | Проверка состояния сервера |
| POST | `/api/v1/projects` | Создание проекта и запуск обработки |
| GET | `/api/v1/projects/{id}/status` | Получение статуса обработки |
| GET | `/api/v1/projects/{id}` | Получение результатов распознавания |
| POST | `/api/v1/projects/{id}/export` | Генерация и скачивание DOCX |

### Статусы обработки

`В очереди` → `Скачивание` → `Конвертация` → `Загрузка в облако` → `Распознавание` → `Готово`

## Как это работает

1. **Загрузка** — пользователь вставляет ссылку Яндекс.Диска на видеоматериал
2. **Скачивание** — бэкенд загружает файл через публичный API Яндекс.Диска
3. **Конвертация** — FFmpeg преобразует видео в одноканальный OPUS-аудиофайл (48 kHz)
4. **Загрузка в S3** — аудио загружается в Yandex Cloud Object Storage
5. **Распознавание** — Yandex SpeechKit выполняет распознавание речи с диаризацией
6. **Верификация** — в веб-интерфейсе отображаются обнаруженные дикторы с процентом эфирного времени и расшифровка с таймкодами; пользователь может скорректировать имена дикторов
7. **Экспорт** — формируется DOCX-документ с отформатированной расшифровкой

## Лицензия

Проприетарный проект. Все права защищены.
