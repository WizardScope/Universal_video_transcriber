#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/WizardScope/Universal_video_transcriber.git}"
UBUNTU_DIR="${UBUNTU_DIR:-/root/uvt_run}"
TERMUX_BIN="$HOME/bin"

printf '\n[UVT] Android / Termux / Ubuntu proot setup\n\n'

pkg update -y
pkg install -y proot-distro git curl

if ! proot-distro list | grep -q '^  ubuntu'; then
  proot-distro install ubuntu
else
  proot-distro install ubuntu 2>/dev/null || true
fi

proot-distro login ubuntu -- bash -lc "
set -e
apt update
apt install -y python3 python3-venv python3-pip ffmpeg git ca-certificates
rm -rf '$UBUNTU_DIR'
git clone --depth 1 '$REPO_URL' '$UBUNTU_DIR'
cd '$UBUNTU_DIR'
python3 -m venv .venv_fw
. .venv_fw/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
python3 - <<'PY'
from pathlib import Path
import re
p = Path('Universal_video_transcriber_v3_6_1.py')
s = p.read_text(encoding='utf-8')
s = re.sub(r'OFFLINE_ONLY\s*=\s*True', 'OFFLINE_ONLY = False', s)
s = re.sub(r'DEVICE_MODE\s*=\s*[\'\"][^\'\"]*[\'\"]', "DEVICE_MODE = 'cpu'", s)
s = re.sub(r'PROFILE\s*=\s*[\'\"][^\'\"]*[\'\"]', "PROFILE = 'fast'", s)
p.write_text(s, encoding='utf-8')
PY
"

mkdir -p "$TERMUX_BIN"

cat > "$TERMUX_BIN/uvt" <<'EOF'
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

INPUT_FILE="${1:-}"

proot-distro login ubuntu -- bash <<'UBUNTU'
set -euo pipefail

cd /root/uvt_run
. .venv_fw/bin/activate

if [ -n "${INPUT_FILE:-}" ]; then
  MEDIA="$INPUT_FILE"
else
  MEDIA=$(find /storage/emulated/0/Download /storage/emulated/0/Music /storage/emulated/0/Movies /storage/emulated/0/DCIM /storage/emulated/0/Pydroid -type f \( -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.mov" -o -iname "*.avi" -o -iname "*.mp3" -o -iname "*.wav" -o -iname "*.m4a" -o -iname "*.ogg" -o -iname "*.webm" \) -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -n 1 | cut -d" " -f2-)
fi

[ -n "$MEDIA" ] || { echo "NO MEDIA FILE FOUND. Put audio/video into Download, Music, Movies or DCIM."; exit 1; }
[ -f "$MEDIA" ] || { echo "MEDIA FILE NOT FOUND: $MEDIA"; exit 1; }

OUT="/storage/emulated/0/Download/UVT_result_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
export MEDIA OUT

python3 - <<'PY'
from pathlib import Path
import os
import re

p = Path('Universal_video_transcriber_v3_6_1.py')
s = p.read_text(encoding='utf-8')
media = os.environ['MEDIA']
out = os.environ['OUT']

s = re.sub(r'OFFLINE_ONLY\s*=\s*True', 'OFFLINE_ONLY = False', s)
s = re.sub(r'DEVICE_MODE\s*=\s*[\'\"][^\'\"]*[\'\"]', "DEVICE_MODE = 'cpu'", s)
s = re.sub(r'PROFILE\s*=\s*[\'\"][^\'\"]*[\'\"]', "PROFILE = 'fast'", s)
s = re.sub(r'INPUT_MEDIA\s*=\s*r?[\'\"].*?[\'\"]', 'INPUT_MEDIA = r' + repr(media), s)
s = re.sub(r'OUTPUT_DIR\s*=\s*r?[\'\"].*?[\'\"]', 'OUTPUT_DIR = r' + repr(out), s)
p.write_text(s, encoding='utf-8')
PY

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

echo "Using media: $MEDIA"
echo "Output folder: $OUT"
python3 Universal_video_transcriber_v3_6_1.py

echo
echo "DONE. Files:"
find "$OUT" -maxdepth 3 -type f
UBUNTU
EOF

cat > "$TERMUX_BIN/uvt-check" <<'EOF'
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

echo "=== TERMUX ==="
command -v proot-distro && echo "proot-distro: OK"
[ -x "$HOME/bin/uvt" ] && echo "uvt: OK" || echo "uvt: NOT FOUND"

proot-distro login ubuntu -- bash <<'UBUNTU'
set -e
cd /root/uvt_run || { echo "uvt_run: NO"; exit 1; }
echo "uvt_run: OK"
[ -f Universal_video_transcriber_v3_6_1.py ] && echo "script: OK" || echo "script: NO"
[ -d .venv_fw ] && echo "venv: OK" || echo "venv: NO"
. .venv_fw/bin/activate
python3 -m py_compile Universal_video_transcriber_v3_6_1.py && echo "syntax: OK"
python3 - <<'PY'
mods = ['av', 'ctranslate2', 'faster_whisper', 'onnxruntime', 'numpy', 'huggingface_hub']
for m in mods:
    try:
        mod = __import__(m)
        print(f'{m}: OK', getattr(mod, '__version__', ''))
    except Exception as e:
        print(f'{m}: BAD -> {e}')
PY
ffmpeg -version 2>/dev/null | head -n 1 || echo "ffmpeg: BAD"
echo "DONE: no transcription started"
UBUNTU
EOF

chmod +x "$TERMUX_BIN/uvt" "$TERMUX_BIN/uvt-check"

grep -qxF 'export PATH="$HOME/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null || echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
grep -qxF 'export PATH="$HOME/bin:$PATH"' "$HOME/.bash_profile" 2>/dev/null || echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bash_profile"
export PATH="$HOME/bin:$PATH"

printf '\n[UVT] Setup complete.\n'
printf 'Run check: uvt-check\n'
printf 'Run transcription: uvt\n'
printf 'Run specific file: uvt "/storage/emulated/0/Download/example.m4a"\n'
