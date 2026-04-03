# -*- coding: utf-8 -*-
"""
v3.6.1 universal

Интересные факты про текущую версию:
- это монолит без внешних постпроцессоров;
- у него есть защита от ASR-зацикливания (бесконечное повторение одного и того же слова/фразы);
- модель универсальна: лекции, подкасты, созвоны, интервью, видеоуроки - всё транскрибирует;
- по умолчанию формирует 3 основных итоговых файла:
    1) *_full_readable.txt  — полный читаемый текст;
    2) *_brief.txt          — краткое содержание по разделам;
    3) *_study_pack.txt     — вопросы, карточки, словарь терминов, план повторения.

Дополнительные технические файлы отключены по умолчанию.

Требования:
    pip install faster-whisper
    pip install av
    ffmpeg / ffprobe в PATH, либо по директории запуска
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import av  # type: ignore
except Exception:
    av = None

from faster_whisper import WhisperModel

try:
    from faster_whisper import BatchedInferencePipeline
except Exception:
    BatchedInferencePipeline = None


# =========================================================
# USER SETTINGS
# =========================================================

INPUT_MEDIA = r"путь откуда взять файл"
OUTPUT_DIR = r"путь куда файл сохранится"
BASE_NAME = ""

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = SCRIPT_DIR / "_hf_models" # модели должны быть скачены в ту же папку, где находится скрипт
OFFLINE_ONLY = True

# "cuda" / "cpu" / "auto"
DEVICE_MODE = "auto"

# "fast" / "balanced" / "quality"
PROFILE = "quality"

# "auto" / "lecture" / "meeting" / "podcast" / "generic"
CONTENT_TYPE = "auto"

LANGUAGE = "ru"          # None = auto detect
TASK = "transcribe"
WORD_TIMESTAMPS = False
USE_VAD = True
MIN_SILENCE_DURATION_MS = 900
CONDITION_ON_PREVIOUS_TEXT = False

QUICK_SAVE_EVERY_SEGMENTS = 300
FULL_SAVE_EVERY_SEGMENTS = 600
PROGRESS_PRINT_EVERY_SEGMENTS = 1

CUSTOM_REPLACEMENTS_PATH = None

# Антизацикливание декодера
ANTI_LOOP_GUARD = True
LOOP_WINDOW_SEGMENTS = 36
LOOP_MIN_DUPLICATES = 10
LOOP_MAX_UNIQUE_NORMALIZED = 4
LOOP_MAX_WORDS_PER_SEGMENT = 8
LOOP_MAX_CHARS_PER_SEGMENT = 80

# По умолчанию создаются 3 основных .txt
SAVE_SRT = False
SAVE_JSON = False
SAVE_DEBUG_JSON = False

# Выходные файлы
SAVE_FULL_READABLE = True
SAVE_BRIEF = True
SAVE_STUDY_PACK = True

# Ограничители для итоговых материалов
MAX_BRIEF_SECTIONS = 14
MAX_KEY_POINTS = 36
MAX_STUDY_QUESTIONS = 24
MAX_FLASHCARDS = 18
MAX_GLOSSARY_TERMS = 18


# =========================================================
# PROFILES
# =========================================================

def profile_settings(profile: str) -> Dict[str, object]:
    profile = profile.lower().strip()

    if profile == "fast":
        return {
            "model_cuda": "turbo",
            "model_cpu": "small",
            "compute_cuda": "int8_float16",
            "compute_cpu": "int8",
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0.0,
            "use_batched": True,
            "batch_size": 4,
        }

    if profile == "quality":
        return {
            "model_cuda": "large-v3",
            "model_cpu": "medium",
            "compute_cuda": "float16", # если компьютер не вывозит - "int8_float16"
            "compute_cpu": "int8",
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "use_batched": False,
            "batch_size": 4,
        }

    return {
        "model_cuda": "turbo",
        "model_cpu": "medium",
        "compute_cuda": "int8_float16",
        "compute_cpu": "int8",
        "beam_size": 3,
        "best_of": 3,
        "temperature": 0.0,
        "use_batched": False,
        "batch_size": 4,
    }


PROFILE_SETTINGS = profile_settings(PROFILE)


# =========================================================
# CONSTANTS
# =========================================================

RUS_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "она", "оно", "они", "на", "я", "с", "со",
    "как", "а", "то", "все", "это", "но", "да", "к", "ко", "у", "же", "вы", "за",
    "бы", "по", "ее", "её", "его", "их", "мы", "от", "до", "из", "или", "ли",
    "для", "о", "об", "про", "при", "над", "под", "без", "так", "тут", "там",
    "еще", "ещё", "уже", "ну", "вот", "где", "когда", "если", "только", "лишь",
    "чтобы", "потому", "поэтому", "который", "которая", "которые", "которое",
    "этот", "эта", "эти", "такой", "такая", "такие", "такое", "один", "одна",
    "одни", "одного", "одном", "два", "три", "раз", "просто", "здесь", "сейчас",
    "будет", "быть", "есть", "нет", "давайте", "значит", "вообще", "очень",
    "самый", "самая", "самое", "самые", "тоже", "либо", "между", "тогда", "потом",
    "дальше", "итак", "ладно", "окей", "ага", "угу", "хорошо", "понятно", "ребят",
    "меня", "вас", "нам", "вам", "нас", "мне", "себе", "ему", "ей", "ними",
    "было", "были", "буду", "можно", "могу", "может", "например", "скажем",
    "соответственно", "принципе", "образом", "этом", "этой", "того", "чем",
    "какой", "какая", "какие", "какое", "каким", "какими", "также", "ровно",
    "сильно", "меньше", "больше", "причем", "причём", "типа", "вроде", "дело",
    "штука", "история", "момент", "штук", "внутри", "снаружи", "алло",
}

GENERIC_KEYWORD_BLACKLIST = {
    "ситуация", "ситуации", "случай", "случаи", "случае", "вариант", "варианты",
    "количество", "время", "нужно", "делать", "сделать", "получить", "получается",
    "получаю", "получится", "видно", "видите", "просто", "вообще", "вопрос",
    "вопросы", "сегодня", "теперь", "дальше", "итак", "ребят", "понятно",
    "история", "штука", "образом", "момент", "данный", "данном", "данная",
    "данные", "потом", "почему", "например", "какая", "какой", "какие",
    "можно", "нельзя", "могу", "может", "будет", "будут", "нету", "есть",
    "сразу", "когда", "чтобы", "этого", "этой", "этот", "такая", "такой",
    "тоже", "ровно", "сильно", "важный", "важно", "нужно", "всего", "самое",
}

SAFE_REPLACEMENTS = {
    "\u00A0": " ",
    "—": " — ",
    "–": " — ",
    "…": "...",
}

# Только безопасные и довольно универсальные замены (можно добавлять свои)
SOFT_REPLACEMENTS = {
    "рок кривая": "ROC-кривая",
    "рок-кривая": "ROC-кривая",
    "рок-аук": "ROC/AUC",
    "аук рок": "AUC-ROC",
    "тру позитив": "true positive",
    "тру негатив": "true negative",
    "фолс позитив": "false positive",
    "фолс негатив": "false negative",
    "пресижн": "precision",
    "рекол": "recall",
    "эйси": "accuracy",
    "эс вм": "SVM",
    "с в м": "SVM",
    "кабли в ближайших соседей": "k-ближайших соседей",
    "keep alive": "keep-alive",
    "head shake": "handshake",
    "headshake": "handshake",
    "web socket": "WebSocket",
    "web rtc": "WebRTC",
    "set cookie": "Set-Cookie",
    "same site": "SameSite",
    "http only": "HttpOnly",
}

TECH_PATTERNS: List[Tuple[str, str]] = [
    (r"\bipv4\b", "IPv4"),
    (r"\bipv6\b", "IPv6"),
    (r"\bcidr\b", "CIDR"),
    (r"\bdns\b", "DNS"),
    (r"\bttl\b", "TTL"),
    (r"\bdhcp\b", "DHCP"),
    (r"\bnat\b", "NAT"),
    (r"\bpat\b", "PAT"),
    (r"\bcg[- ]?nat\b", "CG-NAT"),
    (r"\btcp\b", "TCP"),
    (r"\budp\b", "UDP"),
    (r"\bquic\b", "QUIC"),
    (r"\bhttp\s*1\.\s*1\b", "HTTP 1.1"),
    (r"\bhttp\s*2\b", "HTTP 2"),
    (r"\bhttp\s*3\b", "HTTP 3"),
    (r"\bhttp\b", "HTTP"),
    (r"\bhttps\b", "HTTPS"),
    (r"\btls\s*1\.\s*3\b", "TLS 1.3"),
    (r"\btls\b", "TLS"),
    (r"\bssl\b", "SSL"),
    (r"\baaaa[- ]?запис[ьяеи]+\b", "AAAA-запись"),
    (r"\baaaa\b", "AAAA"),
    (r"\bcname\b", "CNAME"),
    (r"\bmx\b", "MX"),
    (r"\btxt\b", "TXT"),
    (r"\bns\b", "NS"),
    (r"\bsrv\b", "SRV"),
    (r"\bweb[ -]?socket\b", "WebSocket"),
    (r"\bweb[ -]?rtc\b", "WebRTC"),
    (r"\bsse\b", "SSE"),
    (r"\bserver[- ]sent events\b", "Server-Sent Events"),
    (r"\bkeep[- ]?alive\b", "keep-alive"),
    (r"\bhandshake\b", "handshake"),
    (r"\bthree[- ]way[- ]handshake\b", "three-way handshake"),
    (r"\bhead[- ]of[- ]line[- ]blocking\b", "head-of-line blocking"),
    (r"\bjwt\b", "JWT"),
    (r"\bset[- ]cookie\b", "Set-Cookie"),
    (r"\bhttp[- ]only\b", "HttpOnly"),
    (r"\bsame[- ]site\b", "SameSite"),
    (r"\bredis\b", "Redis"),
    (r"\bdtls\b", "DTLS"),
    (r"\bsrtp\b", "SRTP"),
    (r"\bp2p\b", "P2P"),
    (r"\bca\b", "CA"),
    (r"\bcertificate authority\b", "Certificate Authority"),
    (r"\blet'?s encrypt\b", "Let's Encrypt"),
    (r"\bacme\b", "ACME"),
    (r"\broot certificate\b", "Root Certificate"),
    (r"\bwildcard\b", "wildcard"),
    (r"\breverse[- ]proxy\b", "reverse proxy"),
    (r"\brate[- ]limiting\b", "rate limiting"),
    (r"\bround[- ]robin\b", "Round Robin"),
    (r"\bleast connections\b", "Least Connections"),
    (r"\bnginx\b", "Nginx"),
    (r"\bhaproxy\b", "HAProxy"),
    (r"\bcdn\b", "CDN"),
    (r"\bdpi\b", "DPI"),
    (r"\bdeep[- ]packet[- ]inspection\b", "Deep Packet Inspection"),
    (r"\bтспу\b", "ТСПУ"),
]

SECTION_MARKERS = (
    "итак", "теперь", "дальше", "перейдем", "перейдём", "рассмотрим", "следующий",
    "следующая", "следующее", "во-первых", "во-вторых", "во-третьих", "важный момент",
    "обратите внимание", "подведем итог", "подведём итог", "итого", "что такое",
    "начнем", "начнём", "перейдём дальше",
)

IMPORTANT_MARKERS = (
    "важно", "ключев", "обратите внимание", "это означает", "таким образом",
    "следовательно", "позволяет", "недостаток", "преимущество", "критич",
    "главная идея", "если подытожить", "если подвести итог", "итог",
)

SERVICE_PATTERNS = [
    r"подпиш(?:ись|итесь)",
    r"постав(?:ь|ьте) лайк",
    r"не забуд(?:ь|ьте)",
    r"спасибо за просмотр",
    r"до новых встреч",
    r"телеграм[- ]?канал",
    r"спойлер(?:ы)? к новым видео",
    r"если вам понравился контент",
    r"подписывайтесь на канал",
    r"ставьте лайк",
    r"^\s*раз(?:[- ]два(?:[- ]три(?:[- ]четыре(?:[- ]пять)?)?)?)?[.!?]?\s*$",
    r"^\s*меня слышно.*$",
    r"^\s*слышно меня.*$",
    r"^\s*видно презентацию.*$",
    r"^\s*видно экран.*$",
    r"^\s*провер(ка|яем?) (звук|связь|микрофон).*$",
    r"^\s*алло[.!?]?\s*$",
]

SERVICE_SHORT_VOCAB = {
    "слышно", "видно", "алло", "микрофон", "громкость", "окей", "ага", "угу",
    "ребят", "начинаем", "проверка",
}

TITLE_PREFIX_PATTERNS = [
    r"^\s*смысл заключается в том[,]?\s*что\s+",
    r"^\s*и важный момент[,]?\s*",
    r"^\s*важный момент[,]?\s*",
    r"^\s*теперь\s+",
    r"^\s*итак[,]?\s*",
    r"^\s*давайте\s+",
    r"^\s*ну[,]?\s*",
    r"^\s*смотрите[,]?\s*",
]

ABBREV_CANON = {
    "ipv4": "IPv4",
    "ipv6": "IPv6",
    "dns": "DNS",
    "tcp": "TCP",
    "udp": "UDP",
    "http": "HTTP",
    "https": "HTTPS",
    "tls": "TLS",
    "ssl": "SSL",
    "quic": "QUIC",
    "jwt": "JWT",
    "dhcp": "DHCP",
    "nat": "NAT",
    "pat": "PAT",
    "cdn": "CDN",
    "dpi": "DPI",
    "cidr": "CIDR",
    "api": "API",
    "mx": "MX",
    "txt": "TXT",
    "srv": "SRV",
    "ns": "NS",
    "aaaa": "AAAA",
    "www": "www",
    "sse": "SSE",
}


# =========================================================
# HELPERS
# =========================================================

def hhmmss(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"



def srt_time(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    h = ms_total // 3600000
    ms_total %= 3600000
    m = ms_total // 60000
    ms_total %= 60000
    s = ms_total // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"



def slugify(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", "_", name)
    return name or "transcript"



def run_cmd(cmd: List[str]):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)



def get_media_duration_seconds(path: Path) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        cmd = [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        proc = run_cmd(cmd)
        if proc.returncode == 0:
            try:
                return float(proc.stdout.strip())
            except Exception:
                pass

    if av is not None:
        try:
            with av.open(str(path)) as container:
                if container.duration is not None:
                    return float(container.duration / 1_000_000.0)

                best = None
                for stream in list(container.streams.video) + list(container.streams.audio):
                    if stream.duration is not None and stream.time_base is not None:
                        dur = float(stream.duration * stream.time_base)
                        best = dur if best is None else max(best, dur)
                return best
        except Exception:
            return None
    return None



def save_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8-sig", newline="\n")



def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")



def normalize_basic(text: str) -> str:
    for k, v in SAFE_REPLACEMENTS.items():
        text = text.replace(k, v)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    text = re.sub(r"([,.:;!?])(?!\s|$)", r"\1 ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+—\s+", " — ", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"([!?.,;:])\1{1,}", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"^[,.;:!?—\-]+\s*", "", text)
    text = re.sub(r"\s+[,.;:!?—\-]+\s*$", "", text)
    return text.strip()



def normalize_domains_and_tokens(text: str) -> str:
    text = re.sub(r"([A-Za-z0-9_-])\s*\.\s*([A-Za-z0-9_-])", r"\1.\2", text)
    text = re.sub(r"\s*/\s*(\d{1,3})", r"/\1", text)
    text = re.sub(r"\b(HTTP|TLS)\s+(\d)\.\s*(\d)\b", r"\1 \2.\3", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([A-Za-z]+)\s*\.\s*([A-Za-z]{2,})\b", r"\1.\2", text)
    text = re.sub(r"\b([A-Za-z]+)\s*/\s*([A-Za-z0-9]+)\b", r"\1/\2", text)
    return text



def remove_fillers(text: str) -> str:
    patterns = [
        r"\bэ+\b",
        r"\bэ-э+\b",
        r"\bэм+\b",
        r"\bмм+\b",
        r"\bм-м+\b",
        r"\bа-а+\b",
        r"\bсобственно\b",
        r"\bтак сказать\b",
        r"\bкак бы\b",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    text = re.sub(
        r"^\s*(ну|так|вот|окей|ага|короче|соответственно|в принципе|то есть)[, ]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return normalize_basic(text)



def remove_repeated_words(text: str) -> str:
    pattern = re.compile(r"\b([A-Za-zА-Яа-яЁё0-9\-/.+#]{1,60})\b(?:\s+\1\b)+", re.IGNORECASE)
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(r"\1", text)
    return text



def apply_soft_replacements(text: str, replacements: Dict[str, str]) -> str:
    merged = {}
    merged.update(SOFT_REPLACEMENTS)
    merged.update(replacements or {})
    for wrong, correct in merged.items():
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
    return text



def protect_technical_terms(text: str) -> Tuple[str, Dict[str, str]]:
    protected = text
    mapping: Dict[str, str] = {}
    counter = 0

    def repl_factory(canonical: str):
        def _repl(_match):
            nonlocal counter
            token = f"__TECH_{counter}__"
            counter += 1
            mapping[token] = canonical
            return f" {token} "
        return _repl

    for pattern, canonical in TECH_PATTERNS:
        protected = re.sub(pattern, repl_factory(canonical), protected, flags=re.IGNORECASE)

    protected = normalize_basic(protected)
    return protected, mapping



def restore_technical_terms(text: str, mapping: Dict[str, str]) -> str:
    for token, value in mapping.items():
        text = text.replace(token, value)
    return normalize_basic(text)



def smart_capitalize_abbreviations(text: str) -> str:
    for k, v in ABBREV_CANON.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)
    return text



def sentence_case(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]



def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\-/.+#]*", text.lower())



def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9\-/.+#]+", text))



def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", normalize_basic(text))
    return [p.strip() for p in parts if p.strip()]



def load_custom_replacements(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Файл замен не найден: {p}")
        return {}

    rules = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=>" not in line:
            continue
        wrong, correct = line.split("=>", 1)
        wrong = wrong.strip()
        correct = correct.strip()
        if wrong:
            rules[wrong] = correct
    return rules



def looks_like_service_segment(text: str, start_sec: float = 0.0) -> bool:
    t = normalize_basic(text).lower()
    if not t:
        return True

    for pat in SERVICE_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True

    low = t.strip(" .,!?:;")
    if low in {"да", "нет", "ага", "угу", "окей", "хорошо", "отлично", "понятно", "ладно"}:
        return True

    words = tokenize_words(t)
    if len(words) <= 8 and any(w in SERVICE_SHORT_VOCAB for w in words):
        return True

    if start_sec < 10 * 60 and word_count(t) <= 14:
        if any(x in t for x in ("слышно", "видно", "микрофон", "начинаем", "проверка")):
            return True

    return False



def is_too_weak(text: str) -> bool:
    low = normalize_basic(text).lower().strip(" .,!?:;—-")
    if not low:
        return True
    if len(low) <= 2:
        return True
    if re.fullmatch(r"(да|нет|угу|ага|окей|хорошо|понятно|ребят|так|ну|вот|алло|слышно|видно)+", low):
        return True
    return False



def section_marker_score(text: str) -> int:
    low = text.lower()
    score = 0
    for marker in SECTION_MARKERS:
        if low.startswith(marker):
            score += 3
        elif f" {marker} " in f" {low} ":
            score += 1
    if "что такое" in low:
        score += 2
    if "давайте" in low and any(x in low for x in ("разбер", "посмотр", "перейд", "обсуд")):
        score += 2
    return score



def important_score(text: str) -> int:
    low = text.lower()
    score = 0
    for marker in IMPORTANT_MARKERS:
        if marker in low:
            score += 3
    if ":" in text:
        score += 1
    if re.search(r"\b\d+(\.\d+)?\b", text):
        score += 1
    score += min(3, len(set(tokenize_words(text))) // 14)
    return score



def cleanup_title_candidate(text: str) -> str:
    out = text.strip()
    for pat in TITLE_PREFIX_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return normalize_basic(out)



def first_sentence_or_snippet(text: str, limit_words: int = 14) -> str:
    sents = sentence_split(text)
    head = sents[0] if sents else text.strip()
    head = cleanup_title_candidate(head)
    words = head.split()
    if len(words) > limit_words:
        head = " ".join(words[:limit_words]).rstrip(",;:") + "..."
    return head



def title_from_paragraph(text: str) -> str:
    sents = sentence_split(text)
    if not sents:
        return ""

    best_title = ""
    best_score = -10**9
    for sent in sents[:4]:
        cleaned = cleanup_title_candidate(sent)
        if not cleaned:
            continue
        score = important_score(cleaned) + section_marker_score(cleaned) * 2
        if "что такое" in cleaned.lower():
            score += 3
        if len(cleaned.split()) < 4:
            score -= 3
        if looks_like_service_segment(cleaned):
            score -= 10
        if score > best_score:
            best_score = score
            best_title = cleaned

    return first_sentence_or_snippet(best_title or sents[0], 14)



def keyword_candidates(text: str) -> List[str]:
    words = tokenize_words(text)
    result = []
    for w in words:
        if w in RUS_STOPWORDS:
            continue
        if w in GENERIC_KEYWORD_BLACKLIST:
            continue
        if len(w) < 4 or w.isdigit():
            continue
        result.append(w)
    return result



def bigram_candidates(text: str) -> List[str]:
    words = keyword_candidates(text)
    result = []
    for a, b in zip(words, words[1:]):
        if a == b:
            continue
        if a in GENERIC_KEYWORD_BLACKLIST or b in GENERIC_KEYWORD_BLACKLIST:
            continue
        result.append(f"{a} {b}")
    return result



def trigram_candidates(text: str) -> List[str]:
    words = keyword_candidates(text)
    result = []
    for a, b, c in zip(words, words[1:], words[2:]):
        uniq = {a, b, c}
        if len(uniq) < 2:
            continue
        result.append(f"{a} {b} {c}")
    return result



def build_suspicion_flags(raw_text: str, clean_text: str) -> List[str]:
    flags = []
    low_raw = raw_text.lower()

    if raw_text.count("...") >= 2 or "..." in raw_text:
        flags.append("ellipsis_or_cut")
    if re.search(r"\b([A-Za-zА-Яа-яЁё0-9\-/.+#]+)\s+\1\b", raw_text, flags=re.IGNORECASE):
        flags.append("repeated_words")
    if "/ /" in raw_text or ("//" in raw_text and "http" not in low_raw):
        flags.append("slash_pattern")
    if any(x in raw_text for x in ("???", "�")):
        flags.append("encoding_or_unclear")
    if word_count(clean_text) <= 2:
        flags.append("too_short_after_cleaning")
    if not clean_text.strip():
        flags.append("empty_after_cleaning")
    if len(raw_text) > 0 and len(clean_text) / max(len(raw_text), 1) < 0.45:
        flags.append("strong_rewrite_ratio")
    return flags



def clean_segment_text(raw_text: str, custom_replacements: Dict[str, str]) -> Tuple[str, Dict[str, object]]:
    diagnostics: Dict[str, object] = {"raw": raw_text}

    protected, mapping = protect_technical_terms(raw_text)
    stage1 = normalize_basic(protected)
    stage1 = normalize_domains_and_tokens(stage1)
    stage1 = remove_fillers(stage1)
    stage1 = remove_repeated_words(stage1)
    stage1 = apply_soft_replacements(stage1, custom_replacements)
    stage1 = normalize_basic(stage1)

    restored = restore_technical_terms(stage1, mapping)
    restored = smart_capitalize_abbreviations(restored)
    restored = normalize_domains_and_tokens(restored)
    restored = normalize_basic(restored)
    restored = sentence_case(restored)

    diagnostics["protected_terms"] = list(mapping.values())
    diagnostics["stage1"] = stage1
    diagnostics["final"] = restored
    diagnostics["flags"] = build_suspicion_flags(raw_text, restored)
    diagnostics["changes"] = []
    if normalize_basic(raw_text) != stage1:
        diagnostics["changes"].append("stage1_changed")
    if stage1 != restored:
        diagnostics["changes"].append("term_restore_or_stage2_changed")

    return restored, diagnostics



def estimate_eta(start_time: float, processed_seconds: float, total_seconds: Optional[float]):
    elapsed = time.time() - start_time
    if not processed_seconds or not total_seconds or processed_seconds <= 0:
        return None, elapsed
    speed = processed_seconds / elapsed if elapsed > 0 else None
    if not speed or speed <= 0:
        return None, elapsed
    remaining_media = max(0.0, total_seconds - processed_seconds)
    eta = remaining_media / speed
    return eta, elapsed



def print_progress(i: int, seg: Dict[str, object], total_duration: Optional[float], start_time: float):
    if i % PROGRESS_PRINT_EVERY_SEGMENTS != 0:
        return

    processed = float(seg["end"])
    percent = min(100.0, (processed / total_duration) * 100.0) if total_duration else None
    eta, elapsed = estimate_eta(start_time, processed, total_duration)

    preview = str(seg["clean_text"]).replace("\n", " ").strip()
    if len(preview) > 110:
        preview = preview[:110].rstrip() + "..."

    parts = [f"[{i:04d}] {hhmmss(float(seg['start']))} -> {hhmmss(float(seg['end']))}"]
    if percent is not None:
        parts.append(f"{percent:6.2f}%")
    parts.append(preview)
    print(" | ".join(parts))

    if eta is not None and (i % QUICK_SAVE_EVERY_SEGMENTS == 0 or i % FULL_SAVE_EVERY_SEGMENTS == 0):
        print(f"  > elapsed={hhmmss(elapsed)} | eta={hhmmss(eta)}")



def loop_guard_normalize(text: str) -> str:
    text = normalize_basic(text).lower()
    text = re.sub(r"[^a-zа-яё0-9\s-]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_probable_loop_segment(clean_text: str, recent_segments: List[Dict[str, object]]) -> bool:
    if not ANTI_LOOP_GUARD:
        return False

    current = loop_guard_normalize(clean_text)
    if not current:
        return False
    if word_count(current) > LOOP_MAX_WORDS_PER_SEGMENT:
        return False
    if len(current) > LOOP_MAX_CHARS_PER_SEGMENT:
        return False

    recent = [loop_guard_normalize(str(s.get("clean_text", ""))) for s in recent_segments[-LOOP_WINDOW_SEGMENTS:]]
    recent = [x for x in recent if x]
    if len(recent) < LOOP_MIN_DUPLICATES:
        return False

    current_count = sum(1 for x in recent if x == current)
    unique_recent = len(set(recent))

    if current_count >= LOOP_MIN_DUPLICATES and unique_recent <= LOOP_MAX_UNIQUE_NORMALIZED:
        return True

    tail = recent[-12:]
    if len(tail) >= 10 and tail.count(current) >= 8 and len(set(tail)) <= 3:
        return True

    return False

# =========================================================
# RUNTIME
# =========================================================

def detect_runtime() -> Dict[str, str]:
    if DEVICE_MODE == "cpu":
        return {
            "device": "cpu",
            "compute_type": str(PROFILE_SETTINGS["compute_cpu"]),
            "model_size": str(PROFILE_SETTINGS["model_cpu"]),
        }
    return {
        "device": "cuda",
        "compute_type": str(PROFILE_SETTINGS["compute_cuda"]),
        "model_size": str(PROFILE_SETTINGS["model_cuda"]),
    }



def fallback_runtime() -> Dict[str, str]:
    return {
        "device": "cpu",
        "compute_type": str(PROFILE_SETTINGS["compute_cpu"]),
        "model_size": str(PROFILE_SETTINGS["model_cpu"]),
    }



def load_model_with_fallback() -> Tuple[WhisperModel, Dict[str, str], bool]:
    runtime = detect_runtime()
    cpu_threads = max(1, min(6, (os.cpu_count() or 6)))

    try:
        model = WhisperModel(
            runtime["model_size"],
            device=runtime["device"],
            compute_type=runtime["compute_type"],
            cpu_threads=cpu_threads,
            download_root=str(MODEL_CACHE_DIR),
            local_files_only=OFFLINE_ONLY,
        )
        return model, runtime, False
    except Exception as e:
        if DEVICE_MODE == "cpu":
            raise RuntimeError(f"Не удалось запустить CPU-режим: {e}") from e

        fb = fallback_runtime()
        print(f"[WARN] GPU не стартовал ({e}). Переключаюсь на CPU...")
        model = WhisperModel(
            fb["model_size"],
            device=fb["device"],
            compute_type=fb["compute_type"],
            cpu_threads=cpu_threads,
            download_root=str(MODEL_CACHE_DIR),
            local_files_only=OFFLINE_ONLY,
        )
        return model, fb, True


# =========================================================
# CONTENT TYPE + PARAGRAPHS + SECTIONS
# =========================================================

def infer_content_type(segments: List[Dict[str, object]]) -> str:
    if CONTENT_TYPE != "auto":
        return CONTENT_TYPE

    clean_segments = [s for s in segments if str(s.get("clean_text", "")).strip()]
    if not clean_segments:
        return "generic"

    sample = clean_segments[: min(220, len(clean_segments))]
    texts = [str(s["clean_text"]) for s in sample]

    avg_words = sum(word_count(t) for t in texts) / max(len(texts), 1)
    short_ratio = sum(1 for t in texts if word_count(t) <= 6) / max(len(texts), 1)
    question_ratio = sum(1 for t in texts if "?" in t) / max(len(texts), 1)
    marker_ratio = sum(1 for t in texts if section_marker_score(t) >= 2) / max(len(texts), 1)

    if question_ratio >= 0.12 and short_ratio >= 0.28:
        return "meeting"
    if avg_words >= 13 and marker_ratio >= 0.05:
        return "lecture"
    if avg_words >= 10 and short_ratio < 0.24:
        return "podcast"
    return "generic"



def content_layout_settings(content_type: str) -> Dict[str, float]:
    if content_type == "meeting":
        return {
            "paragraph_gap_sec": 1.2,
            "hard_gap_sec": 3.0,
            "min_paragraph_chars": 170,
            "max_paragraph_chars": 900,
            "section_target_sec": 5 * 60,
            "section_min_sec": 90,
            "section_min_paragraphs": 2,
        }
    if content_type == "podcast":
        return {
            "paragraph_gap_sec": 1.6,
            "hard_gap_sec": 4.0,
            "min_paragraph_chars": 240,
            "max_paragraph_chars": 1150,
            "section_target_sec": 8 * 60,
            "section_min_sec": 2 * 60,
            "section_min_paragraphs": 2,
        }
    if content_type == "lecture":
        return {
            "paragraph_gap_sec": 1.8,
            "hard_gap_sec": 4.5,
            "min_paragraph_chars": 280,
            "max_paragraph_chars": 1250,
            "section_target_sec": 8 * 60,
            "section_min_sec": 2 * 60,
            "section_min_paragraphs": 2,
        }
    return {
        "paragraph_gap_sec": 1.5,
        "hard_gap_sec": 3.8,
        "min_paragraph_chars": 220,
        "max_paragraph_chars": 1100,
        "section_target_sec": 7 * 60,
        "section_min_sec": 2 * 60,
        "section_min_paragraphs": 2,
    }



def filtered_segments_for_reading(segments: List[Dict[str, object]]) -> List[Dict[str, object]]:
    result = []
    for s in segments:
        txt = str(s["clean_text"]).strip()
        if not txt:
            continue
        if is_too_weak(txt):
            continue
        if looks_like_service_segment(txt, float(s["start"])):
            continue
        result.append(s)
    return result



def group_into_paragraphs(clean_segments: List[Dict[str, object]], content_type: str) -> List[Dict[str, object]]:
    cfg = content_layout_settings(content_type)
    paragraphs = []
    current: List[Dict[str, object]] = []
    current_chars = 0

    paragraph_gap_sec = float(cfg["paragraph_gap_sec"])
    hard_gap_sec = float(cfg["hard_gap_sec"])
    max_paragraph_chars = int(cfg["max_paragraph_chars"])
    min_paragraph_chars = int(cfg["min_paragraph_chars"])

    def flush():
        nonlocal current, current_chars
        if not current:
            return
        text = " ".join(str(seg["clean_text"]) for seg in current).strip()
        text = normalize_basic(text)
        if text:
            paragraphs.append(
                {
                    "start": float(current[0]["start"]),
                    "end": float(current[-1]["end"]),
                    "text": text,
                    "segments": len(current),
                }
            )
        current = []
        current_chars = 0

    for seg in clean_segments:
        text = str(seg["clean_text"]).strip()
        if not text:
            continue

        start_new = False
        if not current:
            start_new = True
        else:
            prev = current[-1]
            gap = float(seg["start"]) - float(prev["end"])

            if gap >= hard_gap_sec:
                flush()
                start_new = True
            elif gap >= paragraph_gap_sec and current_chars >= min_paragraph_chars:
                flush()
                start_new = True
            elif section_marker_score(text) >= 2 and current_chars >= min_paragraph_chars:
                flush()
                start_new = True
            elif current_chars >= max_paragraph_chars:
                flush()
                start_new = True

        if start_new and not current:
            current.append(seg)
            current_chars = len(text)
        else:
            current.append(seg)
            current_chars += len(text) + 1

    flush()
    return merge_short_paragraphs(paragraphs, content_type)



def merge_short_paragraphs(paragraphs: List[Dict[str, object]], content_type: str) -> List[Dict[str, object]]:
    if not paragraphs:
        return []

    cfg = content_layout_settings(content_type)
    min_paragraph_chars = int(cfg["min_paragraph_chars"])

    merged: List[Dict[str, object]] = []
    for p in paragraphs:
        if not merged:
            merged.append(dict(p))
            continue

        prev = merged[-1]
        gap = float(p["start"]) - float(prev["end"])
        if len(str(p["text"])) < min_paragraph_chars and gap < 5:
            prev["text"] = normalize_basic(str(prev["text"]) + " " + str(p["text"]))
            prev["end"] = p["end"]
            prev["segments"] = int(prev.get("segments", 0)) + int(p.get("segments", 0))
        else:
            merged.append(dict(p))
    return merged



def build_sections(paragraphs: List[Dict[str, object]], content_type: str) -> List[Dict[str, object]]:
    if not paragraphs:
        return []

    cfg = content_layout_settings(content_type)
    target_sec = float(cfg["section_target_sec"])
    min_sec = float(cfg["section_min_sec"])
    min_paragraphs = int(cfg["section_min_paragraphs"])

    sections: List[Dict[str, object]] = []
    bucket: List[Dict[str, object]] = []

    def flush():
        nonlocal bucket
        if not bucket:
            return
        combined = " ".join(str(x["text"]) for x in bucket)
        title = choose_section_title(bucket)
        sections.append(
            {
                "start": float(bucket[0]["start"]),
                "end": float(bucket[-1]["end"]),
                "title": title,
                "text": normalize_basic(combined),
                "paragraphs": [dict(x) for x in bucket],
            }
        )
        bucket = []

    for p in paragraphs:
        if not bucket:
            bucket.append(p)
            continue

        bucket_start = float(bucket[0]["start"])
        bucket_end = float(bucket[-1]["end"])
        current_duration = bucket_end - bucket_start
        p_text = str(p["text"])

        split_now = False
        if section_marker_score(p_text) >= 3 and len(bucket) >= min_paragraphs and current_duration >= min_sec:
            split_now = True
        elif current_duration >= target_sec and len(bucket) >= min_paragraphs:
            split_now = True

        if split_now:
            flush()
        bucket.append(p)

    flush()

    # Если sections получилось слишком много, склеиваются соседние мелкие
    merged: List[Dict[str, object]] = []
    for sec in sections:
        if not merged:
            merged.append(sec)
            continue
        prev = merged[-1]
        prev_duration = float(prev["end"]) - float(prev["start"])
        if prev_duration < min_sec * 0.7 and len(prev["paragraphs"]) < min_paragraphs:
            prev["text"] = normalize_basic(str(prev["text"]) + " " + str(sec["text"]))
            prev["end"] = sec["end"]
            prev["paragraphs"].extend(sec["paragraphs"])
            prev["title"] = choose_section_title(prev["paragraphs"])
        else:
            merged.append(sec)

    return merged[:MAX_BRIEF_SECTIONS] if len(merged) > MAX_BRIEF_SECTIONS else merged



def choose_section_title(paragraphs: List[Dict[str, object]]) -> str:
    if not paragraphs:
        return "Раздел"

    candidates: List[Tuple[int, int, str]] = []
    for idx, p in enumerate(paragraphs[:3]):
        candidate = title_from_paragraph(str(p["text"]))
        if not candidate:
            continue
        score = important_score(candidate) + section_marker_score(candidate) * 2
        score += max(0, 3 - idx)
        if len(candidate.split()) < 3:
            score -= 2
        candidates.append((score, idx, candidate))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][2]

    return first_sentence_or_snippet(str(paragraphs[0]["text"]), 14) or "Раздел"


# =========================================================
# SUMMARY / STUDY PACK
# =========================================================

def paragraph_score_for_highlight(text: str) -> int:
    score = important_score(text) + section_marker_score(text)
    if len(text) > 200:
        score += 1
    if len(text) > 400:
        score += 1
    kws = [w for w in keyword_candidates(text) if w not in GENERIC_KEYWORD_BLACKLIST]
    score += min(4, len(set(kws)) // 4)
    return score



def select_key_points(paragraphs: List[Dict[str, object]], limit: int = MAX_KEY_POINTS) -> List[Tuple[float, str]]:
    scored: List[Tuple[int, float, str]] = []
    for p in paragraphs:
        txt = str(p["text"])
        if looks_like_service_segment(txt, float(p["start"])):
            continue
        title = title_from_paragraph(txt)
        candidate = title or first_sentence_or_snippet(txt, 18)
        if not candidate:
            continue
        score = paragraph_score_for_highlight(txt)
        scored.append((score, float(p["start"]), candidate))

    scored.sort(key=lambda x: (-x[0], x[1]))

    result: List[Tuple[float, str]] = []
    seen = set()
    for _, start, candidate in scored:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append((start, candidate))
        if len(result) >= limit:
            break
    return result



def sentence_for_term(paragraphs: List[Dict[str, object]], term: str) -> Optional[str]:
    pat = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
    for p in paragraphs:
        for sent in sentence_split(str(p["text"])):
            if pat.search(sent):
                return sent
    return None



def extract_glossary(paragraphs: List[Dict[str, object]], max_terms: int = MAX_GLOSSARY_TERMS) -> List[Tuple[str, str]]:
    token_counter = Counter()
    bigram_counter = Counter()
    special_terms = Counter()

    joined = "\n".join(str(p["text"]) for p in paragraphs)

    for p in paragraphs:
        txt = str(p["text"])
        token_counter.update(set(keyword_candidates(txt)))
        bigram_counter.update(set(bigram_candidates(txt)))

        for abbr in re.findall(r"\b[A-ZА-ЯЁ]{2,}(?:[-./][A-ZА-ЯЁ0-9]+)*\b", txt):
            if len(abbr) >= 2:
                special_terms[abbr] += 2

    for _, canonical in TECH_PATTERNS:
        if canonical.lower() in joined.lower():
            special_terms[canonical] += 3

    candidates: List[str] = []
    for term, count in special_terms.most_common(40):
        if count >= 2:
            candidates.append(term)
    for term, count in bigram_counter.most_common(60):
        if count >= 2 and len(term.split()) == 2:
            candidates.append(term)
    for term, count in token_counter.most_common(80):
        if count >= 3:
            candidates.append(term)

    glossary: List[Tuple[str, str]] = []
    seen = set()
    for term in candidates:
        norm = term.lower()
        if norm in seen:
            continue
        seen.add(norm)
        sent = sentence_for_term(paragraphs, term)
        if not sent:
            continue

        definition = sent.strip()
        definition = normalize_basic(definition)

        if len(definition) < 24:
            continue
        glossary.append((term, definition))
        if len(glossary) >= max_terms:
            break

    return glossary



def build_questions(sections: List[Dict[str, object]], glossary: List[Tuple[str, str]], max_questions: int = MAX_STUDY_QUESTIONS) -> List[str]:
    questions: List[str] = []
    seen = set()

    for sec in sections:
        title = str(sec["title"]).strip().rstrip(".?!")
        if title:
            q = f"Что важно понять в теме «{title}»?"
            if q not in seen:
                questions.append(q)
                seen.add(q)
            q2 = f"Как своими словами объяснить тему «{title}»?"
            if q2 not in seen:
                questions.append(q2)
                seen.add(q2)

    for term, definition in glossary:
        q = f"Что означает термин «{term}» и где он используется?"
        if q not in seen:
            questions.append(q)
            seen.add(q)
        if len(questions) >= max_questions:
            break

    return questions[:max_questions]



def build_flashcards(glossary: List[Tuple[str, str]], max_cards: int = MAX_FLASHCARDS) -> List[Tuple[str, str]]:
    cards = []
    for term, definition in glossary[:max_cards]:
        question = f"Что такое {term}?"
        answer = definition
        cards.append((question, answer))
    return cards



def build_revision_plan(sections: List[Dict[str, object]], content_type: str) -> List[str]:
    titles = [str(s["title"]) for s in sections]
    if not titles:
        return ["1) Прочитай полный читаемый файл целиком и выдели 5 ключевых идей."]

    plan = [
        "1) Сначала прочитай весь файл *_full_readable.txt без остановок, чтобы увидеть общую картину.",
        "2) Затем открой *_brief.txt и проверь, понятна ли тебе логика разделов и переходов между ними.",
    ]

    if len(titles) >= 3:
        plan.append(f"3) День 1: повтори разделы 1-{min(3, len(titles))}: " + ", ".join(titles[:3]) + ".")
    if len(titles) >= 6:
        plan.append(f"4) День 2: повтори разделы 4-{min(6, len(titles))}: " + ", ".join(titles[3:6]) + ".")
    elif len(titles) > 3:
        plan.append("4) День 2: повтори оставшиеся разделы и попробуй пересказать их без подглядывания.")
    else:
        plan.append("4) День 2: перечитай сложные места и перескажи содержание своими словами.")

    plan.append("5) После каждого раздела ответь минимум на 2 вопроса из блока «Вопросы для самопроверки».")
    plan.append("6) Карточки используй отдельно: сначала вопрос, затем попытка ответа вслух, потом сверка.")
    plan.append("7) В конце сделай устный пересказ всей темы за 3-5 минут без чтения текста.")

    if content_type == "meeting":
        plan.append("8) Для созвонов и обсуждений отдельно выпиши решения, договорённости и открытые вопросы.")

    return plan


# =========================================================
# BUILD OUTPUT FILES
# =========================================================

def build_full_readable_text(
    paragraphs: List[Dict[str, object]],
    content_type: str,
    runtime: Dict[str, str],
    lang: Optional[str],
    lang_prob: Optional[float],
) -> str:
    lines = []
    lines.append("ПОЛНЫЙ ЧИТАЕМЫЙ ТЕКСТ")
    lines.append("")
    lines.append(f"Тип контента: {content_type}")
    lines.append(f"Модель: {runtime['model_size']} | device={runtime['device']} | compute_type={runtime['compute_type']}")
    if lang is not None and lang_prob is not None:
        lines.append(f"Язык: {lang} (вероятность: {lang_prob:.3f})")
    elif lang is not None:
        lines.append(f"Язык: {lang}")
    lines.append("")

    for p in paragraphs:
        lines.append(f"[{hhmmss(float(p['start']))} - {hhmmss(float(p['end']))}]")
        lines.append(str(p["text"]))
        lines.append("")

    return "\n".join(lines).strip() + "\n"



def build_brief_text(content_type: str, sections: List[Dict[str, object]], key_points: List[Tuple[float, str]]) -> str:
    lines = []
    lines.append("КРАТКОЕ СОДЕРЖАНИЕ")
    lines.append("")
    lines.append(f"Тип контента: {content_type}")
    lines.append("")

    lines.append("РАЗДЕЛЫ")
    lines.append("")
    for i, sec in enumerate(sections, start=1):
        lines.append(f"{i}. [{hhmmss(float(sec['start']))} - {hhmmss(float(sec['end']))}] {sec['title']}")
        sec_paragraphs = sec.get("paragraphs", [])
        teaser_sentences = []
        combined = " ".join(str(x["text"]) for x in sec_paragraphs[:2])
        for sent in sentence_split(combined):
            if len(sent.split()) >= 6 and not looks_like_service_segment(sent):
                teaser_sentences.append(sent)
            if len(teaser_sentences) >= 2:
                break
        for sent in teaser_sentences:
            lines.append(f"   - {sent}")
        lines.append("")

    lines.append("КЛЮЧЕВЫЕ МЫСЛИ")
    lines.append("")
    for start, point in key_points[:MAX_KEY_POINTS]:
        lines.append(f"- [{hhmmss(start)}] {point}")

    return "\n".join(lines).strip() + "\n"



def build_study_pack_text(
    content_type: str,
    sections: List[Dict[str, object]],
    glossary: List[Tuple[str, str]],
    questions: List[str],
    flashcards: List[Tuple[str, str]],
    revision_plan: List[str],
) -> str:
    lines = []
    lines.append("УЧЕБНЫЙ ПАКЕТ / STUDY PACK")
    lines.append("")
    lines.append(f"Тип контента: {content_type}")
    lines.append("")

    lines.append("1. ПЛАН ПОВТОРЕНИЯ")
    lines.append("")
    for item in revision_plan:
        lines.append(item)
    lines.append("")

    lines.append("2. ВОПРОСЫ ДЛЯ САМОПРОВЕРКИ")
    lines.append("")
    for q in questions:
        lines.append(f"- {q}")
    lines.append("")

    lines.append("3. КАРТОЧКИ")
    lines.append("")
    for i, (q, a) in enumerate(flashcards, start=1):
        lines.append(f"Карточка {i}")
        lines.append(f"Вопрос: {q}")
        lines.append(f"Ответ: {a}")
        lines.append("")

    lines.append("4. СЛОВАРЬ ТЕРМИНОВ")
    lines.append("")
    for term, definition in glossary:
        lines.append(f"- {term} — {definition}")
    lines.append("")

    lines.append("5. РАЗДЕЛЫ ДЛЯ ПЕРЕСКАЗА")
    lines.append("")
    for i, sec in enumerate(sections, start=1):
        lines.append(f"{i}. {sec['title']}")
        lines.append(f"   Время: {hhmmss(float(sec['start']))} - {hhmmss(float(sec['end']))}")
    lines.append("")

    return "\n".join(lines).strip() + "\n"



def build_srt(clean_segments: List[Dict[str, object]]) -> str:
    lines = []
    idx = 1
    for seg in clean_segments:
        txt = str(seg["clean_text"]).strip()
        if not txt:
            continue
        if looks_like_service_segment(txt, float(seg["start"])):
            continue
        lines.append(str(idx))
        lines.append(f"{srt_time(float(seg['start']))} --> {srt_time(float(seg['end']))}")
        lines.append(txt)
        lines.append("")
        idx += 1
    return "\n".join(lines).strip() + "\n"


# =========================================================
# SAVE ORCHESTRATION
# =========================================================

def build_json_payload(
    runtime: Dict[str, str],
    used_fallback: bool,
    lang: Optional[str],
    lang_prob: Optional[float],
    total_duration: Optional[float],
    segments: List[Dict[str, object]],
    content_type: str,
    sections: List[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "runtime": runtime,
        "used_cpu_fallback": used_fallback,
        "language": lang,
        "language_probability": lang_prob,
        "total_duration_sec": total_duration,
        "segments_count": len(segments),
        "profile": PROFILE,
        "content_type": content_type,
        "settings": {
            "device_mode": DEVICE_MODE,
            "profile": PROFILE,
            "content_type": CONTENT_TYPE,
            "language": LANGUAGE,
            "task": TASK,
            "use_vad": USE_VAD,
            "min_silence_duration_ms": MIN_SILENCE_DURATION_MS,
            "word_timestamps": WORD_TIMESTAMPS,
            "condition_on_previous_text": CONDITION_ON_PREVIOUS_TEXT,
        },
        "sections": [
            {
                "start": sec["start"],
                "end": sec["end"],
                "title": sec["title"],
            }
            for sec in sections
        ],
        "segments": segments,
    }



def build_debug_payload(segments: List[Dict[str, object]], paragraphs: List[Dict[str, object]], sections: List[Dict[str, object]]) -> Dict[str, object]:
    flag_counter = Counter()
    change_counter = Counter()
    protected_counter = Counter()

    for seg in segments:
        diag = seg.get("diagnostics", {})
        for fl in diag.get("flags", []):
            flag_counter[fl] += 1
        for ch in diag.get("changes", []):
            change_counter[ch] += 1
        for term in diag.get("protected_terms", []):
            protected_counter[term] += 1

    return {
        "summary": {
            "segments_total": len(segments),
            "paragraphs_total": len(paragraphs),
            "sections_total": len(sections),
            "flags": dict(flag_counter),
            "changes": dict(change_counter),
            "protected_terms": dict(protected_counter.most_common(120)),
        },
        "sample_segments": [
            {
                "time": hhmmss(float(seg["start"])),
                "raw": seg.get("text", ""),
                "clean": seg.get("clean_text", ""),
                "flags": seg.get("diagnostics", {}).get("flags", []),
            }
            for seg in segments[:200]
        ],
    }



def generate_outputs(
    out_dir: Path,
    base: str,
    runtime: Dict[str, str],
    used_fallback: bool,
    lang: Optional[str],
    lang_prob: Optional[float],
    total_duration: Optional[float],
    segments: List[Dict[str, object]],
    final: bool = False,
) -> Dict[str, str]:
    reading_segments = filtered_segments_for_reading(segments)
    content_type = infer_content_type(reading_segments or segments)
    paragraphs = group_into_paragraphs(reading_segments, content_type)
    sections = build_sections(paragraphs, content_type)
    key_points = select_key_points(paragraphs, MAX_KEY_POINTS)
    glossary = extract_glossary(paragraphs, MAX_GLOSSARY_TERMS)
    questions = build_questions(sections, glossary, MAX_STUDY_QUESTIONS)
    flashcards = build_flashcards(glossary, MAX_FLASHCARDS)
    revision_plan = build_revision_plan(sections, content_type)

    paths: Dict[str, str] = {}

    full_readable_path = out_dir / f"{base}_full_readable.txt"
    brief_path = out_dir / f"{base}_brief.txt"
    study_pack_path = out_dir / f"{base}_study_pack.txt"
    srt_path = out_dir / f"{base}.srt"
    json_path = out_dir / f"{base}_segments.json"
    debug_json_path = out_dir / f"{base}_debug.json"

    if SAVE_FULL_READABLE:
        save_text(full_readable_path, build_full_readable_text(paragraphs, content_type, runtime, lang, lang_prob))
        paths["FULL_READABLE"] = str(full_readable_path)

    if SAVE_BRIEF:
        save_text(brief_path, build_brief_text(content_type, sections, key_points))
        paths["BRIEF"] = str(brief_path)

    if SAVE_STUDY_PACK:
        save_text(
            study_pack_path,
            build_study_pack_text(content_type, sections, glossary, questions, flashcards, revision_plan),
        )
        paths["STUDY_PACK"] = str(study_pack_path)

    if final and SAVE_SRT:
        save_text(srt_path, build_srt(reading_segments))
        paths["SRT"] = str(srt_path)

    if final and SAVE_JSON:
        save_json(
            json_path,
            build_json_payload(runtime, used_fallback, lang, lang_prob, total_duration, segments, content_type, sections),
        )
        paths["JSON"] = str(json_path)

    if final and SAVE_DEBUG_JSON:
        save_json(debug_json_path, build_debug_payload(segments, paragraphs, sections))
        paths["DEBUG_JSON"] = str(debug_json_path)

    return paths



def quality_warnings(segments: List[Dict[str, object]]) -> List[str]:
    if not segments:
        return ["Нет сегментов в результате транскрибации."]

    suspicious = 0
    short_clean = 0
    service_like = 0
    for seg in segments:
        flags = seg.get("diagnostics", {}).get("flags", [])
        if flags:
            suspicious += 1
        if word_count(str(seg.get("clean_text", ""))) <= 2:
            short_clean += 1
        if looks_like_service_segment(str(seg.get("clean_text", "")), float(seg.get("start", 0.0))):
            service_like += 1

    total = len(segments)
    warnings = []
    suspicious_ratio = suspicious / total
    short_ratio = short_clean / total

    if suspicious_ratio >= 0.15:
        warnings.append(
            "Высокая доля подозрительных сегментов. Для важного материала попробуй PROFILE='quality' или явно задай LANGUAGE."
        )
    if short_ratio >= 0.20:
        warnings.append(
            "Слишком много коротких сегментов после очистки. Возможно, запись шумная или VAD режет слишком агрессивно."
        )
    loop_suspected = sum(1 for seg in segments if "decoder_loop_suspected" in seg.get("diagnostics", {}).get("flags", []))
    if loop_suspected >= 8:
        warnings.append(
            "Сработала защита от ASR-зацикливания: часть повторяющихся сегментов была отброшена. Для этого файла это лучше, чем исходная петля."
        )
    if service_like / total >= 0.15:
        warnings.append(
            "Много служебных/коротких вставок. Это нормально для созвонов, но для лекции может указывать на шумный исходник."
        )
    return warnings


# =========================================================
# MAIN
# =========================================================

def main():
    media_file = Path(INPUT_MEDIA)
    if not media_file.exists():
        raise FileNotFoundError(f"Файл не найден: {media_file}")

    base = BASE_NAME.strip() or media_file.stem
    base = slugify(base)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Определяю длительность медиа...")
    total_duration = get_media_duration_seconds(media_file)
    if total_duration:
        print(f"Длительность: {hhmmss(total_duration)}")
    else:
        print("Длительность определить не удалось.")

    print("Загрузка модели...")
    model, runtime, used_fallback = load_model_with_fallback()
    print(
        f"Режим: device={runtime['device']}, compute_type={runtime['compute_type']}, "
        f"model={runtime['model_size']}, profile={PROFILE}"
    )

    transcribe_target = model
    if bool(PROFILE_SETTINGS["use_batched"]):
        if BatchedInferencePipeline is None:
            print("[WARN] BatchedInferencePipeline недоступен. Использую обычный режим.")
        else:
            transcribe_target = BatchedInferencePipeline(model=model)
            print(f"Пакетный режим включён, batch_size={PROFILE_SETTINGS['batch_size']}")

    kwargs = dict(
        task=TASK,
        beam_size=int(PROFILE_SETTINGS["beam_size"]),
        best_of=int(PROFILE_SETTINGS["best_of"]),
        temperature=float(PROFILE_SETTINGS["temperature"]),
        language=LANGUAGE,
        vad_filter=USE_VAD,
        vad_parameters={"min_silence_duration_ms": MIN_SILENCE_DURATION_MS},
        word_timestamps=WORD_TIMESTAMPS,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
    )
    if bool(PROFILE_SETTINGS["use_batched"]) and BatchedInferencePipeline is not None:
        kwargs["batch_size"] = int(PROFILE_SETTINGS["batch_size"])

    print("Начинаю транскрибацию...")
    start_wall = time.time()
    segments_gen, info = transcribe_target.transcribe(str(media_file), **kwargs)

    lang = getattr(info, "language", None)
    lang_prob = getattr(info, "language_probability", None)
    if lang is not None and lang_prob is not None:
        print(f"Определён язык: {lang} (вероятность: {lang_prob:.3f})")

    custom_replacements = load_custom_replacements(CUSTOM_REPLACEMENTS_PATH)
    collected: List[Dict[str, object]] = []
    last_paths: Dict[str, str] = {}

    for i, seg in enumerate(segments_gen, start=1):
        raw_text = (seg.text or "").strip()
        clean_text, diagnostics = clean_segment_text(raw_text, custom_replacements)

        if is_probable_loop_segment(clean_text, collected):
            flags = diagnostics.setdefault("flags", [])
            if "decoder_loop_suspected" not in flags:
                flags.append("decoder_loop_suspected")
            clean_text = ""

        item = {
            "index": i,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": raw_text,
            "clean_text": clean_text,
            "diagnostics": diagnostics,
        }
        collected.append(item)

        print_progress(i, item, total_duration, start_wall)

        if i % FULL_SAVE_EVERY_SEGMENTS == 0:
            last_paths = generate_outputs(
                out_dir=out_dir,
                base=base,
                runtime=runtime,
                used_fallback=used_fallback,
                lang=lang,
                lang_prob=lang_prob,
                total_duration=total_duration,
                segments=collected,
                final=False,
            )
            print(f"  > Полное сохранение после {i} сегментов")
        elif i % QUICK_SAVE_EVERY_SEGMENTS == 0:
            # Лёгкое сохранение: обновляем только читаемый файл, чтобы была живая промежуточная версия
            if SAVE_FULL_READABLE:
                last_paths = generate_outputs(
                    out_dir=out_dir,
                    base=base,
                    runtime=runtime,
                    used_fallback=used_fallback,
                    lang=lang,
                    lang_prob=lang_prob,
                    total_duration=total_duration,
                    segments=collected,
                    final=False,
                )
            print(f"  > Лёгкое сохранение после {i} сегментов")

    last_paths = generate_outputs(
        out_dir=out_dir,
        base=base,
        runtime=runtime,
        used_fallback=used_fallback,
        lang=lang,
        lang_prob=lang_prob,
        total_duration=total_duration,
        segments=collected,
        final=True,
    )

    total_elapsed = time.time() - start_wall
    warnings = quality_warnings(collected)

    print("\nГотово.")
    print(f"Сегментов: {len(collected)}")
    print(f"Время обработки: {hhmmss(total_elapsed)}")
    for key, value in last_paths.items():
        print(f"{key + ':':16} {value}")

    if warnings:
        print("\nПредупреждения по качеству:")
        for w in warnings:
            print(f"- {w}")


if __name__ == "__main__":
    main()
