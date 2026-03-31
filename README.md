# Universal_video_transcriber

Универсальный Python-скрипт для транскрибации аудио и видео на базе `faster-whisper` с постобработкой текста, защитой от зацикливания декодера и автоматической генерацией итоговых материалов.

## Возможности

- транскрибация аудио и видео;
- поддержка разных типов контента: лекции, подкасты, созвоны, интервью, видеоуроки;
- очистка и нормализация текста после распознавания;
- фильтрация служебных и слабых сегментов;
- защита от повторяющихся зацикленных фрагментов;
- генерация нескольких итоговых файлов для чтения и повторения.

## Что создаётся на выходе

По умолчанию скрипт сохраняет:

- `*_full_readable.txt` — полный читаемый текст;
- `*_brief.txt` — краткое содержание по разделам;
- `*_study_pack.txt` — вопросы, карточки, словарь терминов и план повторения.

## Стек

- Python 3.10+
- `faster-whisper`
- `av`
- `ffmpeg` / `ffprobe`

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/WizardScope/Universal_video_transcriber.git
cd Universal_video_transcriber
```

### 2. Создать и активировать виртуальное окружение

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установить зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Проверить FFmpeg

```bash
ffmpeg -version
ffprobe -version
```

Если команды не находятся, нужно установить FFmpeg и добавить его в `PATH`.

### 5. Указать входной файл и папку для результата

Откройте `Universal_video_transcriber_v3_6_1.py` и заполните в блоке `USER SETTINGS`:

```python
INPUT_MEDIA = r"путь_к_вашему_файлу"
OUTPUT_DIR = r"папка_для_результата"
BASE_NAME = ""
```

### 6. При необходимости настроить режим работы

```python
DEVICE_MODE = "cuda"      # "cuda" / "cpu" / "auto"
PROFILE = "quality"       # "fast" / "balanced" / "quality"
CONTENT_TYPE = "auto"     # "auto" / "lecture" / "meeting" / "podcast" / "generic"
LANGUAGE = "ru"           # None = автоопределение языка
```

Если GPU недоступен или CUDA не настроена, используйте:

```python
DEVICE_MODE = "cpu"
```

### 7. Запуск

```bash
python Universal_video_transcriber_v3_6_1.py
```

## Пример результата

После завершения работы в папке `OUTPUT_DIR` появятся файлы вида:

```text
example_full_readable.txt
example_brief.txt
example_study_pack.txt
```

## Структура репозитория

- `Universal_video_transcriber_v3_6_1.py` — основной скрипт;
- `requirements.txt` — зависимости;
- `README_RU.md` — полная инструкция на русском языке;
- `CHANGELOG.md` — история изменений.

## Замечания

- Скрипт сейчас запускается через редактирование блока `USER SETTINGS` внутри файла.
- Для офлайн-режима используется локальный кэш моделей.
- При недоступности GPU скрипт умеет переключаться на CPU.
- Для длинных файлов есть промежуточные сохранения результата.

## Можно

- вынести настройки в аргументы командной строки;
- добавить конфиг-файл или `.env`;
- добавить примеры входных и выходных данных;
- разбить монолитный файл на модули.

## Лицензия

MIT
