# -*- coding: utf-8 -*-
"""
v3.6.1 universal

Идея версии:
- один монолитный скрипт без внешних постпроцессоров;
- защита от ASR-зацикливания на коротких повторах;
- универсальная логика: лекции, подкасты, созвоны, интервью, видеоуроки;
- 3 основных итоговых файла по умолчанию:
    1) *_full_readable.txt  — полный читаемый текст;
    2) *_brief.txt          — краткое содержание по разделам;
    3) *_study_pack.txt     — вопросы, карточки, словарь терминов, план повторения.

Дополнительные технические файлы отключены по умолчанию.

Требования:
    pip install faster-whisper
Опционально:
    pip install av
    ffmpeg / ffprobe в PATH
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
MODEL_CACHE_DIR = SCRIPT_DIR / "_hf_models"
OFFLINE_ONLY = True

# "cuda" / "cpu" / "auto"
DEVICE_MODE = "cuda"

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

# По умолчанию оставляем только 3 основных txt.
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

# Только безопасные и довольно универсальные замены.
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

    recent = [loop_guard_normalize(str(s.get("clean_text", ""))) for s in recent_segments[-LOOP_WINDOW_SEGMENTS:] if str(s.get("clean_text", "")).strip()]
    if len(recent) < LOOP_MIN_DUPLICATES:
        return False

    duplicates = sum(1 for x in recent if x == current)
    unique_count = len(set(recent))

    if duplicates >= LOOP_MIN_DUPLICATES and unique_count <= LOOP_MAX_UNIQUE_NORMALIZED:
        return True
    return False



def detect_language_from_info(info) -> Tuple[Optional[str], Optional[float]]:
    language = getattr(info, "language", None)
    probability = getattr(info, "language_probability", None)
    try:
        probability = float(probability) if probability is not None else None
    except Exception:
        probability = None
    return language, probability



def infer_content_type(segments: List[Dict[str, object]]) -> str:
    if CONTENT_TYPE != "auto":
        return CONTENT_TYPE

    sample = " ".join(str(s["clean_text"]) for s in segments[:120]).lower()
    if not sample.strip():
        return "generic"

    meeting_markers = ["коллег", "созвон", "повестк", "договорились", "следующий созвон", "вопросы есть"]
    lecture_markers = ["рассмотрим", "определение", "например", "таким образом", "обратите внимание"]
    podcast_markers = ["сегодня поговорим", "в выпуске", "подкаст", "гость", "обсудим"]

    meeting_score = sum(1 for x in meeting_markers if x in sample)
    lecture_score = sum(1 for x in lecture_markers if x in sample)
    podcast_score = sum(1 for x in podcast_markers if x in sample)

    scores = {
        "meeting": meeting_score,
        "lecture": lecture_score,
        "podcast": podcast_score,
        "generic": 0,
    }
    best_type = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_type if scores[best_type] > 0 else "generic"



def filtered_segments_for_reading(segments: List[Dict[str, object]]) -> List[Dict[str, object]]:
    result = []
    for seg in segments:
        text = str(seg["clean_text"]).strip()
        if not text:
            continue
        if is_too_weak(text):
            continue
        if looks_like_service_segment(text, float(seg["start"])):
            continue
        result.append(seg)
    return result



def group_into_paragraphs(clean_segments: List[Dict[str, object]], content_type: str) -> List[Dict[str, object]]:
    if not clean_segments:
        return []

    paragraphs = []
    current = []

    max_gap = 2.8 if content_type in {"lecture", "podcast"} else 2.2
    target_words = 120 if content_type in {"lecture", "podcast"} else 80

    for seg in clean_segments:
        if not current:
            current = [seg]
            continue

        gap = float(seg["start"]) - float(current[-1]["end"])
        cur_words = sum(word_count(str(x["clean_text"])) for x in current)
        marker_boost = section_marker_score(str(seg["clean_text"])) >= 3

        should_break = False
        if gap > max_gap:
            should_break = True
        elif cur_words >= target_words and marker_boost:
            should_break = True
        elif cur_words >= target_words + 35:
            should_break = True

        if should_break:
            text = " ".join(str(x["clean_text"]) for x in current).strip()
            paragraphs.append({
                "start": float(current[0]["start"]),
                "end": float(current[-1]["end"]),
                "text": normalize_basic(text),
                "segments": len(current),
            })
            current = [seg]
        else:
            current.append(seg)

    if current:
        text = " ".join(str(x["clean_text"]) for x in current).strip()
        paragraphs.append({
            "start": float(current[0]["start"]),
            "end": float(current[-1]["end"]),
            "text": normalize_basic(text),
            "segments": len(current),
        })

    return paragraphs



def merge_short_paragraphs(paragraphs: List[Dict[str, object]], content_type: str) -> List[Dict[str, object]]:
    if not paragraphs:
        return []

    min_words = 38 if content_type in {"lecture", "podcast"} else 28
    merged = []

    for p in paragraphs:
        if not merged:
            merged.append(dict(p))
            continue

        if word_count(str(p["text"])) < min_words and section_marker_score(str(p["text"])) == 0:
            merged[-1]["end"] = p["end"]
            merged[-1]["text"] = normalize_basic(str(merged[-1]["text"]) + " " + str(p["text"]))
            merged[-1]["segments"] = int(merged[-1]["segments"]) + int(p["segments"])
        else:
            merged.append(dict(p))

    return merged



def choose_section_title(paragraphs: List[Dict[str, object]]) -> str:
    joined = " ".join(str(p["text"]) for p in paragraphs[:2]).strip()
    title = title_from_paragraph(joined)
    return title or "Раздел"



def build_sections(paragraphs: List[Dict[str, object]], content_type: str) -> List[Dict[str, object]]:
    if not paragraphs:
        return []

    sections = []
    bucket = []
    target_paragraphs = 4 if content_type in {"lecture", "podcast"} else 3

    for p in paragraphs:
        bucket.append(p)
        enough = len(bucket) >= target_paragraphs
        strong_marker = section_marker_score(str(p["text"])) >= 3 and len(bucket) >= 2

        if enough or strong_marker:
            text = "\n\n".join(str(x["text"]) for x in bucket)
            sections.append({
                "start": float(bucket[0]["start"]),
                "end": float(bucket[-1]["end"]),
                "title": choose_section_title(bucket),
                "text": text,
                "paragraphs": list(bucket),
            })
            bucket = []

    if bucket:
        text = "\n\n".join(str(x["text"]) for x in bucket)
        sections.append({
            "start": float(bucket[0]["start"]),
            "end": float(bucket[-1]["end"]),
            "title": choose_section_title(bucket),
            "text": text,
            "paragraphs": list(bucket),
        })

    merged = []
    for sec in sections:
        if not merged:
            merged.append(sec)
            continue
        if len(sec["paragraphs"]) == 1 and word_count(str(sec["text"])) < 55:
            merged[-1]["end"] = sec["end"]
            merged[-1]["text"] = str(merged[-1]["text"]) + "\n\n" + str(sec["text"])
            merged[-1]["paragraphs"].extend(sec["paragraphs"])
            merged[-1]["title"] = choose_section_title(merged[-1]["paragraphs"])
        else:
            merged.append(sec)

    return merged[:MAX_BRIEF_SECTIONS]



def select_key_points(paragraphs: List[Dict[str, object]], limit: int = MAX_KEY_POINTS) -> List[Tuple[float, str]]:
    scored = []
    for p in paragraphs:
        text = str(p["text"])
        score = important_score(text) + section_marker_score(text)
        if looks_like_service_segment(text, float(p["start"])):
            continue
        if word_count(text) < 12:
            continue
        scored.append((score, float(p["start"]), first_sentence_or_snippet(text, 22)))

    scored.sort(key=lambda x: (-x[0], x[1]))

    seen = set()
    result = []
    for _, sec, snippet in scored:
        key = normalize_basic(snippet).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append((sec, snippet))
        if len(result) >= limit:
            break

    result.sort(key=lambda x: x[0])
    return result



def sentence_for_term(paragraphs: List[Dict[str, object]], term: str) -> Optional[str]:
    tl = term.lower()
    for p in paragraphs:
        for sent in sentence_split(str(p["text"])):
            if tl in sent.lower() and 6 <= word_count(sent) <= 36:
                return sent
    return None



def extract_glossary(paragraphs: List[Dict[str, object]], max_terms: int = MAX_GLOSSARY_TERMS) -> List[Tuple[str, str]]:
    text = "\n".join(str(p["text"]) for p in paragraphs)
    counter = Counter()

    for token in keyword_candidates(text):
        counter[token] += 1
    for token in bigram_candidates(text):
        counter[token] += 2
    for token in trigram_candidates(text):
        counter[token] += 3

    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    glossary = []
    used = set()

    for term, _ in ranked:
        norm = term.lower()
        if norm in used:
            continue
        sent = sentence_for_term(paragraphs, term)
        if not sent:
            continue
        definition = first_sentence_or_snippet(sent, 26)
        glossary.append((term, definition))
        used.add(norm)
        if len(glossary) >= max_terms:
            break

    return glossary



def build_questions(sections: List[Dict[str, object]], glossary: List[Tuple[str, str]], max_questions: int = MAX_STUDY_QUESTIONS) -> List[str]:
    questions = []
    seen = set()

    for sec in sections:
        title = str(sec["title"]).rstrip(".")
        q = f"Что имеется в виду в разделе «{title}»?"
        if q not in seen:
            questions.append(q)
            seen.add(q)
        if len(questions) >= max_questions:
            break

    for term, _ in glossary:
        q = f"Как можно объяснить термин «{term}» своими словами?"
        if q not in seen:
            questions.append(q)
            seen.add(q)
        if len(questions) >= max_questions:
            break

    return questions[:max_questions]



def build_flashcards(glossary: List[Tuple[str, str]], max_cards: int = MAX_FLASHCARDS) -> List[Tuple[str, str]]:
    return glossary[:max_cards]



def build_revision_plan(sections: List[Dict[str, object]], content_type: str) -> List[str]:
    plan = []
    if not sections:
        return plan

    titles = [str(sec["title"]) for sec in sections[:6]]
    if content_type == "lecture":
        plan.append("1) Прочитать полный конспект целиком и отметить незнакомые места.")
        plan.append(f"2) Повторить основные разделы: {', '.join(titles)}.")
        plan.append("3) Ответить на вопросы из study pack без подсказок.")
        plan.append("4) Повторить карточки терминов через 1 день и через 3 дня.")
    else:
        plan.append("1) Быстро прочитать краткое содержание и восстановить общую логику материала.")
        plan.append(f"2) Вернуться к ключевым разделам: {', '.join(titles)}.")
        plan.append("3) Проверить себя по вопросам и карточкам.")
        plan.append("4) Через день пересказать материал своими словами по памяти.")
    return plan



def build_full_readable_text(paragraphs: List[Dict[str, object]], content_type: str) -> str:
    lines = []
    header = {
        "lecture": "Полный читаемый конспект",
        "meeting": "Полный читаемый текст созвона",
        "podcast": "Полный читаемый текст выпуска",
        "generic": "Полный читаемый текст",
    }.get(content_type, "Полный читаемый текст")

    lines.append(header)
    lines.append("=" * len(header))
    lines.append("")

    for p in paragraphs:
        lines.append(str(p["text"]))
        lines.append("")

    return "\n".join(lines).strip() + "\n"



def build_brief_text(content_type: str, sections: List[Dict[str, object]], key_points: List[Tuple[float, str]]) -> str:
    title = {
        "lecture": "Краткое содержание лекции",
        "meeting": "Краткое содержание созвона",
        "podcast": "Краткое содержание выпуска",
        "generic": "Краткое содержание",
    }.get(content_type, "Краткое содержание")

    lines = [title, "=" * len(title), ""]

    for i, sec in enumerate(sections, start=1):
        lines.append(f"{i}. {sec['title']}")
        snippet = first_sentence_or_snippet(str(sec["text"]), 28)
        lines.append(f"   {snippet}")
        lines.append(f"   Временной диапазон: {hhmmss(float(sec['start']))} — {hhmmss(float(sec['end']))}")
        lines.append("")

    if key_points:
        lines.append("Ключевые мысли")
        lines.append("-" * len("Ключевые мысли"))
        for _, point in key_points:
            lines.append(f"- {point}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"



def build_study_pack_text(content_type: str, sections: List[Dict[str, object]], glossary: List[Tuple[str, str]], questions: List[str], flashcards: List[Tuple[str, str]], revision_plan: List[str]) -> str:
    title = {
        "lecture": "Study pack по лекции",
        "meeting": "Study pack по созвону",
        "podcast": "Study pack по выпуску",
        "generic": "Study pack",
    }.get(content_type, "Study pack")

    lines = [title, "=" * len(title), ""]

    if questions:
        lines.append("Вопросы для самопроверки")
        lines.append("-" * len("Вопросы для самопроверки"))
        for i, q in enumerate(questions, start=1):
            lines.append(f"{i}. {q}")
        lines.append("")

    if flashcards:
        lines.append("Карточки")
        lines.append("-" * len("Карточки"))
        for i, (front, back) in enumerate(flashcards, start=1):
            lines.append(f"{i}. {front} — {back}")
        lines.append("")

    if glossary:
        lines.append("Словарь терминов")
        lines.append("-" * len("Словарь терминов"))
        for term, definition in glossary:
            lines.append(f"- {term}: {definition}")
        lines.append("")

    if revision_plan:
        lines.append("План повторения")
        lines.append("-" * len("План повторения"))
        lines.extend(revision_plan)
        lines.append("")

    if sections:
        lines.append("Структура материала")
        lines.append("-" * len("Структура материала"))
        for i, sec in enumerate(sections, start=1):
            lines.append(f"{i}. {sec['title']} ({hhmmss(float(sec['start']))} — {hhmmss(float(sec['end']))})")
        lines.append("")

    return "\n".join(lines).strip() + "\n"



def build_srt(clean_segments: List[Dict[str, object]]) -> str:
    blocks = []
    n = 1
    for seg in clean_segments:
        text = str(seg["clean_text"]).strip()
        if not text:
            continue
        blocks.append(str(n))
        blocks.append(f"{srt_time(float(seg['start']))} --> {srt_time(float(seg['end']))}")
        blocks.append(text)
        blocks.append("")
        n += 1
    return "\n".join(blocks).strip() + "\n"



def build_json_payload(runtime: Dict[str, str], used_fallback: bool, lang: Optional[str], lang_prob: Optional[float], total_duration: Optional[float], segments: List[Dict[str, object]], content_type: str, sections: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "runtime": runtime,
        "used_fallback": used_fallback,
        "language": lang,
        "language_probability": lang_prob,
        "total_duration_sec": total_duration,
        "content_type": content_type,
        "segments": segments,
        "sections": [
            {
                "start": float(sec["start"]),
                "end": float(sec["end"]),
                "title": str(sec["title"]),
            }
            for sec in sections
        ],
    }



def build_debug_payload(segments: List[Dict[str, object]], paragraphs: List[Dict[str, object]], sections: List[Dict[str, object]]) -> Dict[str, object]:
    suspicious = []
    for seg in segments:
        flags = list(seg["diagnostics"].get("flags", []))
        if flags:
            suspicious.append({
                "index": seg["index"],
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "raw": str(seg["text"]),
                "clean": str(seg["clean_text"]),
                "flags": flags,
            })

    return {
        "segments_total": len(segments),
        "paragraphs_total": len(paragraphs),
        "sections_total": len(sections),
        "suspicious_segments": suspicious,
    }



def quality_warnings(segments: List[Dict[str, object]]) -> List[str]:
    warnings = []
    if not segments:
        warnings.append("Нет распознанных сегментов.")
        return warnings

    empty_clean = sum(1 for seg in segments if not str(seg["clean_text"]).strip())
    suspicious = 0
    service_like = 0

    for seg in segments:
        flags = list(seg["diagnostics"].get("flags", []))
        if flags:
            suspicious += 1
        if looks_like_service_segment(str(seg["clean_text"]), float(seg["start"])):
            service_like += 1

    total = len(segments)
    if empty_clean / total > 0.35:
        warnings.append("Слишком много пустых сегментов после очистки — возможно, распознавание низкого качества.")
    if suspicious / total > 0.25:
        warnings.append("Много подозрительных сегментов — проверьте исходный звук и язык распознавания.")
    if service_like / total > 0.30:
        warnings.append("Слишком много служебных/технических реплик — итог может быть менее полезным.")

    loop_suspected = sum(1 for seg in segments if "decoder_loop_suspected" in list(seg["diagnostics"].get("flags", [])))
    if loop_suspected > 0:
        warnings.append(f"Обнаружены возможные зацикливания декодера: {loop_suspected} сегм.")

    return warnings



def load_model_with_fallback(device_mode: str, profile: Dict[str, object]):
    tried = []

    def attempt(device: str, model_name: str, compute_type: str):
        tried.append((device, model_name, compute_type))
        print(f"[INFO] Загрузка модели: device={device}, model={model_name}, compute={compute_type}")
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(MODEL_CACHE_DIR),
            local_files_only=OFFLINE_ONLY,
        )
        return model, model_name, device, compute_type

    order = []
    if device_mode == "cuda":
        order.append(("cuda", str(profile["model_cuda"]), str(profile["compute_cuda"])))
        order.append(("cpu", str(profile["model_cpu"]), str(profile["compute_cpu"])))
    elif device_mode == "cpu":
        order.append(("cpu", str(profile["model_cpu"]), str(profile["compute_cpu"])))
    else:
        order.append(("cuda", str(profile["model_cuda"]), str(profile["compute_cuda"])))
        order.append(("cpu", str(profile["model_cpu"]), str(profile["compute_cpu"])))

    last_error = None
    for device, model_name, compute_type in order:
        try:
            return attempt(device, model_name, compute_type)
        except Exception as e:
            last_error = e
            print(f"[WARN] Не удалось загрузить модель ({device}, {model_name}, {compute_type}): {e}")

    raise RuntimeError(f"Не удалось загрузить модель. Попытки: {tried}. Последняя ошибка: {last_error}")



def transcribe_media(model, media_path: Path, profile: Dict[str, object]):
    beam_size = int(profile["beam_size"])
    best_of = int(profile["best_of"])
    temperature = float(profile["temperature"])
    use_batched = bool(profile["use_batched"])
    batch_size = int(profile["batch_size"])

    vad_parameters = {"min_silence_duration_ms": MIN_SILENCE_DURATION_MS}

    kwargs = {
        "language": LANGUAGE,
        "task": TASK,
        "log_progress": False,
        "beam_size": beam_size,
        "best_of": best_of,
        "patience": 1.0,
        "length_penalty": 1.0,
        "repetition_penalty": 1.02,
        "no_repeat_ngram_size": 4,
        "temperature": temperature,
        "compression_ratio_threshold": 2.5,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": CONDITION_ON_PREVIOUS_TEXT,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": not WORD_TIMESTAMPS,
        "max_initial_timestamp": 1.0,
        "word_timestamps": WORD_TIMESTAMPS,
        "prepend_punctuations": '"\'“¿([{-',
        "append_punctuations": '"\'.。,，!！?？:：”)]}、',
        "multilingual": True,
        "vad_filter": USE_VAD,
        "vad_parameters": vad_parameters,
        "max_new_tokens": None,
        "chunk_length": None,
        "clip_timestamps": "0",
        "hallucination_silence_threshold": None,
        "hotwords": None,
        "language_detection_threshold": 0.5,
        "language_detection_segments": 1,
    }

    if use_batched and BatchedInferencePipeline is not None:
        pipeline = BatchedInferencePipeline(model=model)
        return pipeline.transcribe(str(media_path), batch_size=batch_size, **kwargs)
    return model.transcribe(str(media_path), **kwargs)



def generate_outputs(out_dir: Path, base: str, runtime: Dict[str, str], used_fallback: bool, lang: Optional[str], lang_prob: Optional[float], total_duration: Optional[float], segments: List[Dict[str, object]], final: bool = False) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered = filtered_segments_for_reading(segments)
    content_type = infer_content_type(filtered)
    paragraphs = merge_short_paragraphs(group_into_paragraphs(filtered, content_type), content_type)
    sections = build_sections(paragraphs, content_type)
    key_points = select_key_points(paragraphs)
    glossary = extract_glossary(paragraphs)
    questions = build_questions(sections, glossary)
    flashcards = build_flashcards(glossary)
    revision_plan = build_revision_plan(sections, content_type)

    paths = {}

    if SAVE_FULL_READABLE:
        full_text = build_full_readable_text(paragraphs, content_type)
        p = out_dir / f"{base}_full_readable.txt"
        save_text(p, full_text)
        paths["full_readable"] = str(p)

    if SAVE_BRIEF:
        brief_text = build_brief_text(content_type, sections, key_points)
        p = out_dir / f"{base}_brief.txt"
        save_text(p, brief_text)
        paths["brief"] = str(p)

    if SAVE_STUDY_PACK:
        study_text = build_study_pack_text(content_type, sections, glossary, questions, flashcards, revision_plan)
        p = out_dir / f"{base}_study_pack.txt"
        save_text(p, study_text)
        paths["study_pack"] = str(p)

    if SAVE_SRT and final:
        srt = build_srt(filtered)
        p = out_dir / f"{base}.srt"
        save_text(p, srt)
        paths["srt"] = str(p)

    if SAVE_JSON and final:
        payload = build_json_payload(runtime, used_fallback, lang, lang_prob, total_duration, segments, content_type, sections)
        p = out_dir / f"{base}.json"
        save_json(p, payload)
        paths["json"] = str(p)

    if SAVE_DEBUG_JSON and final:
        payload = build_debug_payload(segments, paragraphs, sections)
        p = out_dir / f"{base}_debug.json"
        save_json(p, payload)
        paths["debug_json"] = str(p)

    warnings = quality_warnings(segments)
    if warnings:
        p = out_dir / f"{base}_warnings.txt"
        save_text(p, "\n".join(f"- {w}" for w in warnings) + "\n")
        paths["warnings"] = str(p)

    return paths



def main():
    media_path = Path(INPUT_MEDIA)
    if not media_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {media_path}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = BASE_NAME.strip() or slugify(media_path.stem)
    custom_replacements = load_custom_replacements(CUSTOM_REPLACEMENTS_PATH)
    total_duration = get_media_duration_seconds(media_path)

    model, model_name, device_name, compute_type = load_model_with_fallback(DEVICE_MODE, PROFILE_SETTINGS)
    used_fallback = DEVICE_MODE == "cuda" and device_name == "cpu"

    start_time = time.time()
    segments_gen, info = transcribe_media(model, media_path, PROFILE_SETTINGS)
    lang, lang_prob = detect_language_from_info(info)

    runtime = {
        "device": device_name,
        "compute_type": compute_type,
        "model": model_name,
        "profile": PROFILE,
        "content_type": CONTENT_TYPE,
        "language": str(lang),
    }

    collected = []
    last_paths = {}

    for i, seg in enumerate(segments_gen, start=1):
        raw_text = (seg.text or "").strip()
        clean_text, diagnostics = clean_segment_text(raw_text, custom_replacements)

        if is_probable_loop_segment(clean_text, collected):
            diagnostics.setdefault("flags", []).append("decoder_loop_suspected")
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
        print_progress(i, item, total_duration, start_time)

        if i % QUICK_SAVE_EVERY_SEGMENTS == 0:
            last_paths = generate_outputs(out_dir, base, runtime, used_fallback, lang, lang_prob, total_duration, collected, final=False)
            print(f"[INFO] Промежуточное сохранение: {i} сегментов")

        if i % FULL_SAVE_EVERY_SEGMENTS == 0:
            last_paths = generate_outputs(out_dir, base, runtime, used_fallback, lang, lang_prob, total_duration, collected, final=False)
            print(f"[INFO] Полное промежуточное сохранение: {i} сегментов")

    final_paths = generate_outputs(out_dir, base, runtime, used_fallback, lang, lang_prob, total_duration, collected, final=True)
    elapsed = time.time() - start_time

    print("\nГотово.")
    print(f"Файл: {media_path}")
    print(f"Длительность медиа: {hhmmss(total_duration) if total_duration else 'неизвестно'}")
    print(f"Время обработки: {hhmmss(elapsed)}")
    print(f"Язык: {lang} ({lang_prob:.3f})" if lang_prob is not None else f"Язык: {lang}")
    print(f"Профиль: {PROFILE} | устройство={device_name} | модель={model_name} | compute={compute_type}")
    if used_fallback:
        print("[INFO] Использован fallback на CPU.")

    print("\nСохранённые файлы:")
    for _, p in (final_paths or last_paths).items():
        print(f" - {p}")


if __name__ == "__main__":
    main()
