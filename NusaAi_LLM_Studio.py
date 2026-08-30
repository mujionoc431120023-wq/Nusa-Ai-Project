#!/usr/bin/env python3
"""
Nusa Ai LLM Studio
Fitur: Server inferensi lokal, chat, training, dataset download, MCP,
dukungan model HF/local, dan pilihan runtime backend (PyTorch/ONNX/OpenVINO/GGUF).
"""

# Skrip ini bergaya dinamis (PyTorch/transformers/Flask/FastAPI/Tkinter) sehingga
# family diagnostik "unknown type" dari strict mode Pylance sebagian besar adalah
# false positive. Aturan berikut dimatikan khusus untuk file ini; aturan error
# yang bermakna (undefined, possibly-unbound, attribute-access, call-issue, dst.)
# tetap aktif dan diperbaiki di sumbernya.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportUntypedFunctionDecorator=false, reportUnnecessaryIsInstance=false, reportIndexIssue=false, reportUnusedImport=false, reportWildcardImportFromLibrary=false

import os
import sys
import json
import time
import threading
import subprocess
import traceback
import queue
import re
import ast
import operator
import importlib.util  # dipakai aturan instalasi dependensi
from typing import TYPE_CHECKING, Any, Callable, Set

try:
    import torch
except ImportError:
    torch = None
    print("WARNING: PyTorch tidak ditemukan. Jalankan tombol 'Auto Install Dependencies' di aplikasi.")

# Informasi CUDA hanya dicetak saat file dijalankan sebagai aplikasi utama.
# Saat modul ini diimpor oleh library lain (mis. Nusa_Ai_cli.py), cek GPU
# dilakukan oleh pihak pemanggil sehingga import tetap cepat dan bersih.
if torch is not None and __name__ == "__main__":
    print("CUDA Available:", torch.cuda.is_available())
    print("PyTorch CUDA Version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU Device Name:", torch.cuda.get_device_name(0))

# Perbesar timeout koneksi HuggingFace menjadi 10 menit (600 detik)
os.environ["HF_HUB_ETAG_TIMEOUT"] = "600"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
# ================== DETEKSI GPU AMD/INTEL (DIRECTML) ==================
def _dml_onnx_provider_available() -> bool:
    """True jika onnxruntime sudah memiliki DmlExecutionProvider (GPU AMD/Intel via DirectX)."""
    try:
        import onnxruntime as _ort  # type: ignore[import-untyped]
        return "DmlExecutionProvider" in _ort.get_available_providers()
    except Exception:
        return False

def detect_directml_devices() -> "list[str]":
    """Daftar GPU AMD/Intel yang siap dipakai via DirectML untuk RX 560 XT dkk.

    Dua jalur:
      1) torch-directml   -> cocok untuk runtime backend "pytorch"
      2) onnxruntime DML  -> cocok untuk runtime backend "onnx"
    """
    found = []

    # Jalur 1: torch-directml (hanya berfungsi bila versinya cocok dengan torch)
    try:
        import torch_directml  # type: ignore[import-untyped]
        _cnt = torch_directml.device_count() if hasattr(torch_directml, "device_count") else 1
        for i in range(_cnt):
            try:
                _name = torch_directml.device_name(i)
            except Exception:
                _name = f"GPU DirectML #{i}"
            found.append(f"[torch-directml] {_name}")
    except Exception:
        pass

    # Jalur 2: ONNX Runtime DmlExecutionProvider
    if _dml_onnx_provider_available():
        try:
            import onnxruntime as _ort  # type: ignore[import-untyped]
            _opt = _ort.SessionOptions()
            found.append(
                "[onnxruntime] DmlExecutionProvider aktif (pakai Runtime Backend = onnx "
                "untuk GPU AMD)."
            )
        except Exception:
            pass

    return found

# ================== TKINTER (DI-IMPOR SECARA LAZY) ==================
# Modul ini diimpor sebagai pustaka oleh Nusa_Ai_cli.py (atau skrip lain),
# sehingga Tkinter hanya dibutuhkan saat mode GUI. Import Tkinter dilakukan
# oleh _import_gui() yang dipanggil dari main(). Dengan begitu, modul dapat
# diimpor tanpa Tkinter dan tanpa menyebabkan sys.exit dari proses CLI.
GUI_AVAILABLE = False

# Nama Tkinter di-injeksi ke namespace global saat runtime oleh _import_gui()
# (import lazy agar modul tetap bisa dipakai CLI tanpa Tkinter). Blok TYPE_CHECKING
# di bawah TIDAK dieksekusi saat runtime; ia hanya memberitahu type-checker
# (Pylance/mypy) bahwa nama-nama ini valid di module scope.
if TYPE_CHECKING:
    from tkinter import (
        BooleanVar, Button, Canvas, Checkbutton, DoubleVar, Entry, Frame,
        IntVar, Label, Listbox, PhotoImage, Scale, Spinbox, StringVar, Text,
        Tk, filedialog, messagebox, scrolledtext, ttk,
    )
    # Konstanta Tk (END, W, N, S, E, LEFT, RIGHT, TOP, BOTTOM, BOTH, WORD, X, Y, ...)
    from tkinter.constants import *  # noqa: F401,F403


def _import_gui() -> bool:
    """Impor Tkinter beserta nama yang dipakai GUI ke namespace global.

    Mengembalikan True bila sukses, False bila Tkinter tidak tersedia.
    """
    global GUI_AVAILABLE
    if GUI_AVAILABLE:
        return True
    try:
        import tkinter as _tk
        from tkinter import ttk, scrolledtext, filedialog, messagebox
    except ImportError:
        GUI_AVAILABLE = False  # type: ignore[constant-readonly]
        return False

    # Kelas widget & variabel yang dipakai oleh seluruh kode GUI
    # (setara dengan `from tkinter import *` untuk nama yang diperlukan).
    _gui_names = {
        "Tk": _tk.Tk, "IntVar": _tk.IntVar, "DoubleVar": _tk.DoubleVar,
        "StringVar": _tk.StringVar, "BooleanVar": _tk.BooleanVar,
        "PhotoImage": _tk.PhotoImage, "Listbox": _tk.Listbox, "Text": _tk.Text,
        "Frame": _tk.Frame, "Button": _tk.Button, "Label": _tk.Label,
        "Checkbutton": _tk.Checkbutton, "Scale": _tk.Scale,
        "Spinbox": _tk.Spinbox, "Entry": _tk.Entry, "Canvas": _tk.Canvas,
    }
    globals().update(_gui_names)

    # Konstanta Tk (END, W, NORMAL, LEFT, BOTH, WORD, X, Y, ...)
    try:
        import tkinter.constants as _tkconst
        for _n in dir(_tkconst):
            if not _n.startswith("_"):
                globals()[_n] = getattr(_tkconst, _n)
    except Exception:
        pass

    globals().update(ttk=ttk, scrolledtext=scrolledtext,
                     filedialog=filedialog, messagebox=messagebox)
    GUI_AVAILABLE = True  # type: ignore[constant-readonly]
    return True


# ================== OPTIONAL IMPORTS ==================
try:
    import requests
except ImportError:
    requests = None

try:
    import mcp
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
    MCP_AVAILABLE = True  # type: ignore[constant-readonly]
except ImportError:
    # Fallback agar nama selalu terdefinisi di module scope; pemakaian hanya
    # diizinkan saat MCP_AVAILABLE bernilai True (guarded di _run_mcp_server).
    mcp = None  # type: ignore[assignment]
    Server = None  # type: ignore[assignment]
    Tool = None  # type: ignore[assignment]
    TextContent = None  # type: ignore[assignment]
    MCP_AVAILABLE = False  # type: ignore[constant-readonly]

# ================== DAFTAR MODEL PRESET ==================
MODEL_PRESETS = {
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma-2-2B-it": "google/gemma-2-2b-it",
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Claude (API - tidak lokal)": "ANTHROPIC_API",
}

# ================== PRESET MODEL CODING AGENT ==================
CODING_AGENT_PRESETS = {
    "Coder-Qwen2.5-Coder-0.5B-Instruct (ringan)": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "Coder-Qwen2.5-Coder-1.5B-Instruct": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Coder-Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Coder-CodeGemma-2b-it": "google/codegemma-2b-it",
}

# ================== INFO APLIKASI & VERSI ==================
APP_NAME = "Nusa Ai LLM Studio"
APP_VERSION = "1.0.0"

# ================== ROOT SINKRONISASI (SATU SERVER) ==================
def _is_frozen() -> bool:
    """True jika aplikasi dibungkus PyInstaller (exe)."""
    return bool(getattr(sys, "frozen", False))


def resource_path(rel: str) -> str:
    """Path aset yang benar dalam dua kondisi:
      - skrip langsung (dev): relatif terhadap folder skrip
      - exe PyInstaller     : relatif ke folder _MEIPASS (aset di-bundle)
    """
    if _is_frozen():
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel)


if _is_frozen():
    # Saat instalasi, data (models/, nusa_ai/, config/, data/) berada di
    # FOLDER INSTALASI (folder exe), bukan di _MEIPASS (extract sementara onefile).
    STUDIO_DIR = os.path.dirname(os.path.abspath(sys.executable))  # type: ignore[constant-readonly]
else:
    STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))  # type: ignore[constant-readonly]
NUSA_AI_DIR = os.path.join(STUDIO_DIR, "nusa_ai")
HOME_DIR = os.path.expanduser("~")

# ================== LOGO / IKON RESMI ==================
# Dev: icons/ di samping skrip. Exe: di-bundle ke _MEIPASS/icons oleh PyInstaller.
_ifrozen = _is_frozen()
ICON_DIR = os.path.join(STUDIO_DIR, "icons")
if _ifrozen:
    _meipass = getattr(sys, "_MEIPASS", ICON_DIR)
    _bundle_icons = os.path.join(_meipass, "icons")
    if os.path.isdir(_bundle_icons):
        ICON_DIR = _bundle_icons  # type: ignore[constant-readonly]
LOGO_ICO = os.path.join(ICON_DIR, "logo.ico")
LOGO_PNG = os.path.join(ICON_DIR, "logo.png")
LOGO_GIF = os.path.join(ICON_DIR, "logo.gif")

# Semua folder model yang dipindai & digabung ke SATU registry preset:
#   1. <project>/models                  -> hasil training & model umum
#   2. <project>/nusa_ai/models          -> model earth_ai + hub LM Studio (GGUF)
#   3. ~/.lmstudio/models                -> model LM Studio lokal
#   4. ~/.lmstudio/hub/models            -> hub LM Studio (mis. Bonsai-27B-GGUF)
#   5. nusa_ai/extensions/models         -> model dari folder extensions
LOCAL_MODELS_DIRS = [
    os.path.join(STUDIO_DIR, "models"),
    os.path.join(NUSA_AI_DIR, "models"),
    os.path.join(NUSA_AI_DIR, "extensions", "models"),
    os.path.join(HOME_DIR, ".lmstudio", "models"),
    os.path.join(HOME_DIR, ".lmstudio", "hub", "models"),
]
LOCAL_MODELS_DIR = LOCAL_MODELS_DIRS[0]  # direktori utama output training

# Folder data pelatihan: prioritaskan nusa_ai/data, fallback ke <project>/data
DATA_DIRS = [
    os.path.join(NUSA_AI_DIR, "data"),
    os.path.join(STUDIO_DIR, "data"),
]
DEFAULT_DATA_DIR = next((d for d in DATA_DIRS if os.path.isdir(d)), DATA_DIRS[0])

# Awalan label preset untuk model yang ditemukan secara lokal
LOCAL_MODEL_PREFIX = "[Lokal] "

# Kata kunci file GGUF yang BUKAN model utama (multimodal projector / LoRA)
GGUF_EXCLUDE_KEYWORDS = ("mmproj", "-lora", "adapter", "iquest")

# File bobot yang menandakan folder adalah model Transformers utuh
# (folder GGUF ala LM Studio juga punya config.json tapi TIDAK punya bobot ini)
TRANSFORMERS_WEIGHT_FILES = (
    "model.safetensors", "pytorch_model.bin", "model.ckpt", "tf_model.h5",
    "model.safetensors.index.json", "pytorch_model.bin.index.json",
)

# ================== BONSAI AI (AGENT LOKAL) ==================
# Bonsai AI (prism-ml/bonsai-27b, GGUF) dipakai sebagai model agent.
# Cek lokasi kandidat dari yang paling diprioritaskan.
BONSAI_GGUF_CANDIDATES = [
    os.path.join(NUSA_AI_DIR, "models", "prism-ml", "lmstudio-community",
                 "Bonsai-27B-GGUF", "Bonsai-27B-Q1_0.gguf"),
    os.path.join(HOME_DIR, ".lmstudio", "hub", "models", "prism-ml",
                 "lmstudio-community", "Bonsai-27B-GGUF", "Bonsai-27B-Q1_0.gguf"),
]
BONSAI_PRESET_NAME = "Bonsai-AI-27B (Agent Lokal GGUF)"


def find_bonsai_gguf() -> "str | None":
    """Cari file GGUF Bonsai AI di lokasi kandidat yang dikenal."""
    for p in BONSAI_GGUF_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def register_bonsai_agent_preset() -> "str | None":
    """Daftarkan Bonsai AI ke registry preset sebagai model agent (jika GGUF ada)."""
    path = find_bonsai_gguf()
    if path:
        MODEL_PRESETS[BONSAI_PRESET_NAME] = path
        CODING_AGENT_PRESETS[BONSAI_PRESET_NAME] = path
    return path


def _has_transformers_weights(files: "list[str]") -> bool:
    """True jika folder berisi bobot model Transformers (bukan repo GGUF saja)."""
    names = set(files)
    if any(w in names for w in TRANSFORMERS_WEIGHT_FILES):
        return True
    # Bobot shard (model-00001-of-xxx.safetensors / pytorch_model-00001.bin)
    return any(f.endswith((".safetensors", ".bin")) for f in files
               if not f.lower().endswith(".gguf"))


def _root_label(root_dir: str) -> str:
    """Label unik per folder sumber agar nama preset antar-folder tidak bertabrakan."""
    for base in (STUDIO_DIR, HOME_DIR):
        try:
            rel = os.path.relpath(root_dir, base)
        except ValueError:
            continue
        if not rel.startswith(".."):
            return rel
    return os.path.basename(root_dir)

SUPPORTED_CODE_EXT = (
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go",
    ".rs", ".php", ".rb", ".sh", ".bat", ".ps1", ".sql", ".html", ".css",
)

MODEL_SYNC_CONFIG_FILE = os.path.join(STUDIO_DIR, "config", "model_sync.json")


def scan_local_models(log_fn: "Callable[[str], None] | None" = None) -> "dict[str, str]":
    """Pindai SEMUA folder model (LOCAL_MODELS_DIRS) secara rekursif dan
    kembalikan dict {nama_preset: path_lokal} untuk SATU registry server.

    Mendukung:
      - folder berisi config.json  -> model Transformers lokal
      - folder berisi adapter_config.json -> model hasil LoRA/PEFT
      - file .gguf (termasuk bersarang ala hub LM Studio) -> llama.cpp
      - file .gguf multimodal (mmproj-*) dan LoRA otomatis dilewati
    """
    found = {}
    for root_dir in LOCAL_MODELS_DIRS:
        if not os.path.isdir(root_dir):
            if log_fn:
                log_fn(f"[Sync Model] Folder tidak ada, dilewati: {root_dir}")
            continue

        count_before = len(found)
        label = _root_label(root_dir)
        for root, dirs, files in os.walk(root_dir):
            # Hindari folder internal / cache yang tidak relevan (sejak awal walk)
            dirs[:] = [d for d in dirs
                       if not d.startswith(".") and d not in ("__pycache__", "blobs", "refs")]

            # Folder model Transformers / PEFT — HANYA jika punya file bobot.
            # Folder GGUF ala LM Studio juga berisi config.json tetapi tanpa
            # bobot safetensors/bin, jadi harus tetap dipindai GGUF-nya.
            if ("config.json" in files or "adapter_config.json" in files) and _has_transformers_weights(files):
                rel = os.path.relpath(root, root_dir)
                name = f"{LOCAL_MODEL_PREFIX}{label}/{rel}"
                if name not in found:
                    found[name] = root

            # Cari file GGUF (bisa bersarang beberapa level, termasuk di folder
            # yang juga berisi config.json seperti repo model LM Studio)
            for fname in sorted(files):
                if not fname.lower().endswith(".gguf"):
                    continue
                low = fname.lower()
                if any(k in low for k in GGUF_EXCLUDE_KEYWORDS):
                    continue
                rel = os.path.relpath(os.path.join(root, fname), root_dir)
                name = f"{LOCAL_MODEL_PREFIX}{label}/{rel}"
                if name not in found:
                    found[name] = os.path.join(root, fname)

        if log_fn:
            log_fn(f"[Sync Model] {len(found) - count_before} model dari {root_dir}")

    if log_fn:
        log_fn(f"[Sync Model] TOTAL {len(found)} model lokal tersinkron ke registry")
    return found


def sync_model_presets(log_fn: "Callable[[str], None] | None" = None) -> "dict[str, str]":
    """Gabungkan model lokal dari folder models/ ke dalam MODEL_PRESETS."""
    local_models = scan_local_models(log_fn=log_fn)
    for name, path in local_models.items():
        MODEL_PRESETS[name] = path
    return dict(MODEL_PRESETS)


def save_model_sync_config(model_id: str) -> None:
    """Simpan riwayat sinkronisasi/training model ke config/model_sync.json."""
    try:
        cfg = {}
        if os.path.exists(MODEL_SYNC_CONFIG_FILE):
            with open(MODEL_SYNC_CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        cfg["last_synced_model"] = model_id
        cfg["synced_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        history = cfg.get("history", [])
        if not history or history[-1].get("model") != model_id:
            history.append({"model": model_id, "time": cfg["synced_at"]})
        cfg["history"] = history[-20:]
        os.makedirs(os.path.dirname(MODEL_SYNC_CONFIG_FILE), exist_ok=True)
        with open(MODEL_SYNC_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"WARNING: gagal menyimpan konfigurasi sinkronisasi: {e}")


def load_model_sync_config() -> "dict[str, Any]":
    """Muat konfigurasi sinkronisasi jika ada."""
    try:
        if os.path.exists(MODEL_SYNC_CONFIG_FILE):
            with open(MODEL_SYNC_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ================== SINKRONISASI EXTENSIONS (BACKENDS/FRAMEWORKS/PLUGINS) ==================
# Struktur folder nusa_ai/extensions (kompatibel LM Studio):
#   extensions/backends        -> runtime llama.cpp (llama-server.exe) varian GPU/CPU
#                                 (nvidia-cuda / amd-rocm / vulkan / avx2 CPU-only,
#                                  versi 2.20.1, 2.28.2, 2.29.1)
#   extensions/backends/vendor -> DLL vendor (CUDA/ROCm/Vulkan) per varian backend
#   extensions/frameworks      -> framework (harmony, lmlink-connector)
#   extensions/plugins         -> plugin lmstudio (js-code-sandbox, rag-v1)
#                                 dan mcp (brave-search, context7, github, dst.)
#   extensions/models          -> folder model tambahan (ikut registry preset lokal)
NUSA_AI_EXTENSIONS_DIR = os.path.join(NUSA_AI_DIR, "extensions")
EXT_BACKENDS_DIR = os.path.join(NUSA_AI_EXTENSIONS_DIR, "backends")
EXT_VENDOR_DIR = os.path.join(EXT_BACKENDS_DIR, "vendor")
EXT_FRAMEWORKS_DIR = os.path.join(NUSA_AI_EXTENSIONS_DIR, "frameworks")
EXT_PLUGINS_DIR = os.path.join(NUSA_AI_EXTENSIONS_DIR, "plugins")
EXT_MODELS_DIR = os.path.join(NUSA_AI_EXTENSIONS_DIR, "models")

# Sub-folder vendor yang ikut dimasukkan ke PATH/DLL directory (mis. rocm/bin)
VENDOR_PATH_SUBDIRS = ("", "bin", "lib")

# Preferensi varian GPU backend per pilihan device (urutan = prioritas fallback).
# None berarti varian CPU-only (backend tanpa entri "gpu" di manifest).
DEVICE_BACKEND_PREFERENCE = {
    "cuda": ("CUDA", "Vulkan", None),
    "directml": ("ROCm", "Vulkan", None),   # DirectML umumnya dipakai GPU AMD
    "vulkan": ("Vulkan", None),
    "auto": ("CUDA", "ROCm", "Vulkan", None),
    "cpu": (None,),
}


def parse_version_tuple(version: str) -> "tuple[int, ...]":
    """'2.29.1' -> (2, 29, 1) agar versi bisa dibandingkan/diurutkan."""
    try:
        return tuple(int(x) for x in str(version).split("."))
    except (ValueError, AttributeError):
        return (0,)


def load_backend_manifest(backend_dir: str) -> "dict[str, Any] | None":
    """Baca backend-manifest.json dari folder extension (backend/framework)."""
    manifest_path = os.path.join(backend_dir, "backend-manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def discover_llama_cpp_backends(log_fn: "Callable[[str], None] | None" = None) -> "list[dict[str, Any]]":
    """Pindai extensions/backends dan kembalikan daftar runtime llama.cpp
    (llama-server.exe) yang terdeteksi, terurut versi terbaru lebih dulu.

    Sumber data: backend-manifest.json + engine-protocol-server-artifacts.json
    di tiap folder backend (mis. llama.cpp-win-x86_64-nvidia-cuda-avx2-2.29.1).

    Setiap entri berisi: dir, exe, name, version, gpu_framework, gpu_make,
    dan vendor_dirs (folder DLL vendor: win-llama-cuda-vendor-v2,
    win-llama-rocm-vendor-v6, win-llama-vulkan-vendor-v2).
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)

    backends = []
    if not os.path.isdir(EXT_BACKENDS_DIR):
        _log(f"[Ext] Folder backend tidak ditemukan: {EXT_BACKENDS_DIR}")
        return backends

    for entry in sorted(os.listdir(EXT_BACKENDS_DIR)):
        backend_dir = os.path.join(EXT_BACKENDS_DIR, entry)
        if entry == "vendor" or not os.path.isdir(backend_dir):
            continue
        manifest = load_backend_manifest(backend_dir)
        if not manifest or manifest.get("engine") != "llama.cpp":
            continue

        server_info = manifest.get("engine_protocol_server") or {}
        exe_rel = server_info.get("executable_relative_path", "llama-server.exe")
        exe_path = os.path.join(backend_dir, exe_rel)
        if not os.path.isfile(exe_path):
            _log(f"[Ext] Backend dilewati (llama-server.exe tidak ada): {entry}")
            continue

        gpu = manifest.get("gpu") or {}
        # Folder vendor terkait sesuai manifest (mis. win-llama-cuda-vendor-v2)
        vendor_dirs = []
        for pkg in manifest.get("vendor_lib_package_names", []):
            vdir = os.path.join(EXT_VENDOR_DIR, pkg)
            if os.path.isdir(vdir):
                vendor_dirs.append(vdir)

        backends.append({
            "dir": backend_dir,
            "exe": exe_path,
            "name": manifest.get("name", entry),
            "version": manifest.get("version", "0"),
            "gpu_framework": gpu.get("framework"),
            "gpu_make": gpu.get("make"),
            "vendor_dirs": vendor_dirs,
        })
        _log(f"[Ext] Backend llama.cpp terdeteksi: {manifest.get('name', entry)} "
             f"v{manifest.get('version', '?')} "
             f"(GPU: {gpu.get('framework') or 'CPU-only'})")

    backends.sort(key=lambda b: (parse_version_tuple(b["version"]),
                                 b["gpu_framework"] or ""), reverse=True)
    _log(f"[Ext] TOTAL {len(backends)} backend llama.cpp tersinkron "
         f"dari extensions/backends")
    return backends


def pick_llama_cpp_backend(device: str, backends: "list[dict[str, Any]] | None" = None, log_fn: "Callable[[str], None] | None" = None) -> "dict[str, Any] | None":
    """Pilih backend llama.cpp terbaik untuk device yang dipilih.

    Urutan fallback mengikuti DEVICE_BACKEND_PREFERENCE; di dalam varian
    GPU yang sama dipilih versi tertinggi (daftar sudah terurut turun).
    """
    if backends is None:
        backends = discover_llama_cpp_backends(log_fn=log_fn)
    if not backends:
        return None
    preference = DEVICE_BACKEND_PREFERENCE.get(device, DEVICE_BACKEND_PREFERENCE["auto"])
    for framework in preference:
        candidates = [b for b in backends if (b["gpu_framework"] or None) == framework]
        if candidates:
            return candidates[0]
    # Fallback terakhir: varian CPU-only (paling kompatibel)
    cpu_only = [b for b in backends if not b["gpu_framework"]]
    return cpu_only[0] if cpu_only else backends[0]


def prepare_backend_environment(backend: "dict[str, Any] | None") -> "dict[str, str] | None":
    """Siapkan environment (PATH + DLL directory) agar llama-server.exe dapat
    memuat DLL vendor (CUDA/ROCm/Vulkan) dari extensions/backends/vendor."""
    if not backend:
        return None
    search_dirs = [backend["dir"]]
    for vdir in backend.get("vendor_dirs", []):
        for sub in VENDOR_PATH_SUBDIRS:
            p = os.path.join(vdir, sub) if sub else vdir
            if os.path.isdir(p):
                search_dirs.append(p)

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(search_dirs + [env.get("PATH", "")])
    if os.name == "nt":
        for p in search_dirs:
            try:
                os.add_dll_directory(p)
            except (OSError, AttributeError):
                pass
    return env


def discover_extension_frameworks(log_fn: "Callable[[str], None] | None" = None) -> "list[dict[str, Any]]":
    """Sinkronkan daftar framework dari extensions/frameworks
    (harmony-win-x86_64-avx2, lmlink-connector-win-x86_64-avx2)."""
    frameworks = []
    if not os.path.isdir(EXT_FRAMEWORKS_DIR):
        return frameworks
    for entry in sorted(os.listdir(EXT_FRAMEWORKS_DIR)):
        fdir = os.path.join(EXT_FRAMEWORKS_DIR, entry)
        if not os.path.isdir(fdir):
            continue
        manifest = load_backend_manifest(fdir)
        if not manifest:
            continue
        frameworks.append({
            "dir": fdir,
            "name": manifest.get("name", entry),
            "framework": manifest.get("framework", entry),
            "version": manifest.get("version", "?"),
            "domains": manifest.get("domains", []),
        })
        if log_fn:
            log_fn(f"[Ext] Framework terdeteksi: {manifest.get('name', entry)} "
                   f"v{manifest.get('version', '?')} "
                   f"domains={manifest.get('domains', [])}")
    return frameworks


def discover_extension_plugins(log_fn: "Callable[[str], None] | None" = None) -> "list[dict[str, Any]]":
    """Sinkronkan daftar plugin dari extensions/plugins:
      - lmstudio: js-code-sandbox, rag-v1
      - mcp: brave-search, context7, github, local-postgres, playwright
    """
    plugins = []
    if not os.path.isdir(EXT_PLUGINS_DIR):
        return plugins
    for owner in sorted(os.listdir(EXT_PLUGINS_DIR)):
        owner_dir = os.path.join(EXT_PLUGINS_DIR, owner)
        if not os.path.isdir(owner_dir):
            continue
        for entry in sorted(os.listdir(owner_dir)):
            pdir = os.path.join(owner_dir, entry)
            manifest_path = os.path.join(pdir, "manifest.json")
            if not os.path.isdir(pdir) or not os.path.isfile(manifest_path):
                continue
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except (OSError, json.JSONDecodeError):
                manifest = {}
            plugins.append({
                "dir": pdir,
                "owner": manifest.get("owner", owner),
                "name": manifest.get("name", entry),
                "runner": manifest.get("runner", "?"),
            })
            if log_fn:
                log_fn(f"[Ext] Plugin terdeteksi: {manifest.get('owner', owner)}/"
                       f"{manifest.get('name', entry)} "
                       f"(runner: {manifest.get('runner', '?')})")
    return plugins

# ================== SINKRONISASI NUSAAI_CODEASSIST (EXTENSION MODUL) ==================
# Folder NusaAi_codeassist/ di samping server utama berisi modul platform
# "nusa_ai Code Assist" (ekstensi editor VSIX: package.json + dist/extension.js
# + agent/ + webview/). Server utama men-sinkronkannya ke SATU platform:
#   1) Terdeteksi & didaftarkan saat inisialisasi GUI/CLI (registry extensions)
#   2) Config bridge (config/codeassist_bridge.json) ditulis agar modul tahu
#      endpoint server inferensi Nusa Ai (REST /generate, /health, /)
#   3) Info modul tampil di GUI (log + tombol "Sync CodeAssist") dan CLI
#      (python Nusa_Ai_cli.py codeassist)
CODEASSIST_DIR = os.path.join(STUDIO_DIR, "NusaAi_codeassist")
CODEASSIST_PACKAGE_FILE = os.path.join(CODEASSIST_DIR, "package.json")
CODEASSIST_MANIFEST_FILE = os.path.join(CODEASSIST_DIR, ".vsixmanifest")
CODEASSIST_BRIDGE_CONFIG_FILE = os.path.join(STUDIO_DIR, "config",
                                             "codeassist_bridge.json")
CODEASSIST_MODULE_NAME = "nusa_ai Code Assist"


def discover_codeassist_extension(log_fn=None):
    """Deteksi modul NusaAi_codeassist dan baca metadata inisialisasinya.

    Mengembalikan dict info (atau None bila modul tidak ada):
      name, displayName, version, publisher, main, agent_dir, webview_dir,
      a2a_setting (nama setting VS Code untuk alamat agent server)
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)

    if not os.path.isdir(CODEASSIST_DIR):
        _log(f"[CodeAssist] Modul tidak ditemukan: {CODEASSIST_DIR}")
        return None

    info = {
        "dir": CODEASSIST_DIR,
        "name": CODEASSIST_MODULE_NAME,
        "displayName": CODEASSIST_MODULE_NAME,
        "version": "?",
        "publisher": "nusa_ai",
        "main": "./dist/extension.js",
        "agent_dir": os.path.join(CODEASSIST_DIR, "agent"),
        "webview_dir": os.path.join(CODEASSIST_DIR, "webview"),
        "a2a_setting": "nusa_aicodeassist.a2a.address",
    }

    try:
        with open(CODEASSIST_PACKAGE_FILE, "r", encoding="utf-8") as f:
            pkg = json.load(f)
        info["name"] = pkg.get("name", info["name"])
        info["displayName"] = pkg.get("displayName", info["displayName"])
        info["version"] = pkg.get("version", info["version"])
        info["publisher"] = pkg.get("publisher", info["publisher"])
        info["main"] = pkg.get("main", info["main"])
    except (OSError, json.JSONDecodeError) as e:
        _log(f"[CodeAssist] WARNING package.json tidak terbaca: {e}")

    _log(f"[CodeAssist] Modul terdeteksi: {info['displayName']} "
         f"v{info['version']} (publisher: {info['publisher']}) -> {CODEASSIST_DIR}")
    return info


def sync_codeassist_extension(port=8000, model=None, server_url=None, log_fn=None):
    """Sinkronkan modul NusaAi_codeassist ke server utama Nusa Ai.

    Menulis config bridge (config/codeassist_bridge.json) berisi endpoint
    server inferensi lokal sehingga ekstensi Code Assist terhubung ke SATU
    server utama. Kembalikan path config bridge yang ditulis.
    """
    info = discover_codeassist_extension(log_fn=log_fn)
    if info is None:
        raise FileNotFoundError(
            f"Modul NusaAi_codeassist tidak ditemukan di: {CODEASSIST_DIR}")

    base = (server_url or f"http://localhost:{int(port)}").rstrip("/")
    bridge = {
        "module": info["name"],
        "display_name": info["displayName"],
        "version": info["version"],
        "publisher": info["publisher"],
        "module_dir": CODEASSIST_DIR,
        "server_url": base,
        "endpoints": {
            "generate": f"{base}/generate",
            "health": f"{base}/health",
            "index": f"{base}/",
        },
        "active_model": model or load_model_sync_config().get("last_synced_model") or "",
        "vscode_settings": {
            # Setting ekstensi untuk alamat agent (A2A) mengarah ke server Nusa Ai
            info["a2a_setting"]: base,
        },
        "synced_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(CODEASSIST_BRIDGE_CONFIG_FILE), exist_ok=True)
    with open(CODEASSIST_BRIDGE_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(bridge, f, indent=2, ensure_ascii=False)
    if log_fn:
        log_fn(f"[CodeAssist] Bridge tersinkron: {CODEASSIST_BRIDGE_CONFIG_FILE} "
               f"(server: {base})")
    return CODEASSIST_BRIDGE_CONFIG_FILE


def load_codeassist_bridge_config():
    """Muat config bridge CodeAssist jika ada."""
    try:
        if os.path.exists(CODEASSIST_BRIDGE_CONFIG_FILE):
            with open(CODEASSIST_BRIDGE_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ================== BUILDER DATASET CODING AGENT ==================
def _load_structured_samples(obj):
    """Ekstrak sampel (instruction, input, output) dari objek JSON terstruktur."""
    samples = []
    if isinstance(obj, list):
        for item in obj:
            samples.extend(_load_structured_samples(item))
        return samples
    if not isinstance(obj, dict):
        return samples

    # Format Alpaca: {"instruction", "input", "output"}
    if "instruction" in obj and ("output" in obj or "response" in obj):
        samples.append((
            str(obj["instruction"]),
            str(obj.get("input", "") or ""),
            str(obj.get("output", obj.get("response", "")) or ""),
        ))
        return samples

    # Format chat: {"messages": [{"role","content"}, ...]}
    if "messages" in obj and isinstance(obj["messages"], list):
        instruction, answer = [], []
        for msg in obj["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                continue
            if role == "user":
                instruction.append(content)
            elif role == "assistant":
                answer.append(content)
        if instruction and answer:
            samples.append((" ".join(instruction), "", "\n\n".join(answer)))
        return samples

    return samples


def _extract_code_blocks_from_markdown(text, source_name, max_chars=6000):
    """Ambil pasangan (konteks, blok kode) dari dokumen markdown/text."""
    pairs = []
    fence_pattern = re.compile(r"```([\w+\-]*)\n(.*?)```", re.DOTALL)
    matches = list(fence_pattern.finditer(text))
    for idx, m in enumerate(matches):
        lang = (m.group(1) or "").strip() or "text"
        code = m.group(2).strip()
        if len(code.splitlines()) < 3:
            continue  # terlalu pendek untuk dilatih

        # Konteks = teks non-kode sebelum blok ini (heading/paragraf)
        prev_end = matches[idx - 1].end() if idx > 0 else 0
        context_raw = text[prev_end:m.start()]
        context_lines = [ln.strip("# ").strip() for ln in context_raw.strip().splitlines()]
        context_lines = [ln for ln in context_lines if ln]
        base = os.path.splitext(os.path.basename(source_name))[0]
        context = " ".join(context_lines[-3:])[:400] if context_lines else base

        instruction = (
            f"Anda adalah Nusa Ai Coding Agent. Berdasarkan konteks berikut tentang "
            f"'{context}', tuliskan dan jelaskan kode {lang} yang sesuai:"
        )
        output = f"Berikut kode {lang} beserta penjelasannya:\n\n```{lang}\n{code}\n```\n"
        if len(output) > max_chars:
            output = output[:max_chars] + "\n... (dipotong)"
        pairs.append((instruction, "", output))
    return pairs


def build_coding_agent_dataset(data_dir, output_path, log_fn=None,
                               max_samples=50000, max_chunk_chars=12000,
                               include_code_ext=True):
    """Bangun dataset pelatihan Coding Agent dari seluruh file di folder data/.

    Mendukung sumber data:
      - .jsonl : baris per objek (Alpaca / chat messages / text)
      - .json  : array atau dict (Alpaca / chat messages)
      - .md/.txt : ekstraksi blok kode + konteks menjadi instruksi
      - .py/.js/.c/dll  : keseluruhan file menjadi sampel kode

    Output: file JSONL format Alpaca {"instruction","input","output"}.
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)

    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Folder data tidak ditemukan: {data_dir}")

    samples = []          # list of (instruction, input, output)
    seen_hashes = set()

    def add_sample(instr, inp, outp):
        key = hash((instr[:200], outp[:200]))
        if key in seen_hashes:
            return False
        seen_hashes.add(key)
        samples.append((instr.strip(), inp.strip(), outp.strip()))
        return True

    skipped_binary_ext = {".npz", ".parquet", ".keras", ".h5", ".pkl",
                          ".zip", ".gz", ".bin", ".safetensors", ".png",
                          ".jpg", ".jpeg", ".ico", ".exe", ".dll"}

    n_files = 0
    for root, _dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            full = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext in skipped_binary_ext:
                continue
            try:
                size_mb = os.path.getsize(full) / (1024 * 1024)
                if size_mb > 300:  # lewati file raksasa demi keamanan memori
                    _log(f"[Dataset] Lewati file terlalu besar ({size_mb:.0f} MB): {fname}")
                    continue
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read(max_chunk_chars * 400)
            except Exception as e:
                _log(f"[Dataset] Gagal membaca {fname}: {e}")
                continue

            before = len(samples)

            if ext == ".jsonl":
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    structured = _load_structured_samples(obj)
                    if structured:
                        for s in structured:
                            add_sample(*s)
                    elif isinstance(obj, dict) and ("text" in obj or "content" in obj):
                        body = obj.get("text") or obj.get("content")
                        for p in _extract_code_blocks_from_markdown(body, fname):
                            add_sample(*p)
            elif ext == ".json":
                obj = None
                try:
                    if size_mb <= 280:
                        # Parse utuh agar tidak ada sampel yang hilang karena pemotongan
                        with open(full, "r", encoding="utf-8", errors="ignore") as jf:
                            obj = json.load(jf)
                except (json.JSONDecodeError, OSError):
                    obj = None
                except MemoryError:
                    _log(f"[Dataset] Memori tidak cukup untuk {fname}, dilewati.")
                    obj = None
                if obj is not None:
                    for s in _load_structured_samples(obj):
                        add_sample(*s)
                else:
                    # Bukan JSON valid / terlalu besar, perlakukan sebagai teks
                    for p in _extract_code_blocks_from_markdown(raw, fname):
                        add_sample(*p)
            elif ext in (".md", ".txt"):
                for p in _extract_code_blocks_from_markdown(raw, fname):
                    add_sample(*p)
            elif include_code_ext and ext in SUPPORTED_CODE_EXT:
                lines = raw.splitlines()
                if len(lines) >= 5:
                    lang = ext.lstrip(".")
                    instr = (
                        f"Anda adalah Nusa Ai Coding Agent. Tulis ulang dan jelaskan "
                        f"kode {lang} pada file '{fname}':"
                    )
                    code_out = raw
                    if len(code_out) > max_chunk_chars:
                        code_out = code_out[:max_chunk_chars] + "\n... (dipotong)"
                    add_sample(instr, "", f"```{lang}\n{code_out}\n```")

            if len(samples) > before:
                n_files += 1
                _log(f"[Dataset] {fname}: +{len(samples) - before} sampel")

            if len(samples) >= max_samples:
                _log("[Dataset] Batas maksimum sampel tercapai.")
                break
        if len(samples) >= max_samples:
            break

    if not samples:
        raise ValueError("Tidak ada sampel coding yang bisa diekstrak dari folder data.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for instr, inp, outp in samples:
            f.write(json.dumps({
                "instruction": instr, "input": inp, "output": outp,
            }, ensure_ascii=False) + "\n")

    _log(f"[Dataset] SELESAI: {len(samples)} sampel dari {n_files} file disimpan di:")
    _log(f"[Dataset] {output_path}")
    return output_path



# ================== TOOLS & UTILITAS AGENT CHAT ==================
from html import unescape as _html_unescape

CHAT_SKILLS = ["Creativity", "Empathy", "Logic", "Critical Thinking",
               "Curiosity", "Adaptability", "Emotional Intelligence"]
CHAT_MAX_FILE_CHARS = 6000      # batas karakter per file dalam konteks
CHAT_MAX_TREE_FILES = 150       # batas entri struktur folder
CHAT_MAX_FOLDER_FILES = 6       # batas file yang dibaca dari folder proyek
CHAT_SKIP_DIRS = {".git", "__pycache__", "node_modules", "venv", ".venv",
                  ".llmstudio", "blobs", "refs", ".idea", ".vscode",
                  "dist", "build", "__MACOSX"}

CHAT_HELP_TEXT = (
    "Perintah chat yang tersedia:\n"
    "  /help           - tampilkan bantuan ini\n"
    "  /clear          - bersihkan riwayat chat\n"
    "  /save           - ekspor percakapan ke file Markdown\n"
    "  /calc <expr>    - kalkulator (mis. /calc 2*(3+4)^2)\n"
    "  /search <query> - pencarian web via DuckDuckGo\n"
    "  /tree           - tampilkan struktur folder proyek terlampir\n"
    "  /sysinfo        - info sistem (OS/CPU/RAM/GPU)\n"
    "  /files          - daftar file/folder terlampir\n"
    "  /skills         - tampilkan nilai skill agent aktif\n\n"
    "Tips:\n"
    "- Add Files / Add Folder Proyek menyematkan konteks ke prompt agent.\n"
    "- Mode Agent: system prompt coding agent + skill fokus + tools.\n"
    "- Pesan berawalan 'cari ...' otomatis memicu web search (bila tool aktif).\n"
    "- Pesan berupa hitungan (mis. 12*(3+4)) otomatis dihitung tool kalkulator."
)

# Operator aritmatika yang diizinkan kalkulator aman
_SAFE_MATH_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def tool_calculator(expression):
    """Evaluasi ekspresi aritmatika secara aman (AST, tanpa eval mentah)."""
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_MATH_BINOPS:
            return _SAFE_MATH_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            val = _eval(node.operand)
            return -val if isinstance(node.op, ast.USub) else val
        raise ValueError("ekspresi tidak didukung (hanya angka + - * / % ^)")

    # Terima '^' sebagai pangkat (gaya kalkulator) sebelum di-parse sebagai Python
    tree = ast.parse(expression.strip().replace("^", "**"), mode="eval")
    result = _eval(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression.strip()} = {result}"


def tool_web_search(query, max_results=5, timeout=20):
    """Pencarian web via DuckDuckGo (HTML/lite) tanpa API key."""
    if requests is None:
        raise RuntimeError("Library 'requests' tidak tersedia.")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    html = None
    for endpoint, method in (("https://html.duckduckgo.com/html/", "POST"),
                             ("https://lite.duckduckgo.com/lite/", "GET")):
        try:
            if method == "POST":
                r = requests.post(endpoint, data={"q": query}, headers=headers,
                                  timeout=timeout)
            else:
                r = requests.get(endpoint, params={"q": query}, headers=headers,
                                 timeout=timeout)
            if r.status_code == 200 and "result" in r.text:
                html = r.text
                break
        except requests.RequestException:
            continue
    if html is None:
        raise RuntimeError("tidak bisa mengakses DuckDuckGo (cek koneksi internet)")

    if "result__a" in html:  # endpoint html
        raw = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                         html, re.DOTALL)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|div)>',
                              html, re.DOTALL)
    else:  # endpoint lite
        raw = re.findall(r'<a rel="nofollow" href="([^"]+)"[^>]*>(.*?)</a>',
                         html, re.DOTALL)
        snippets = re.findall(r'class="result-snippet"[^>]*>(.*?)</td>', html, re.DOTALL)

    def _clean(s):
        return _html_unescape(re.sub(r"<[^>]+>", "", s)).strip()

    results = []
    for i, (href, title) in enumerate(raw[:max_results]):
        url = href
        if "uddg=" in url:  # redirect DuckDuckGo -> URL asli
            try:
                from urllib.parse import unquote, urlparse, parse_qs
                url = parse_qs(urlparse(url).query).get("uddg", [url])[0]
                url = unquote(url)
            except Exception:
                pass
        snippet = _clean(snippets[i])[:250] if i < len(snippets) else ""
        results.append(f"{i + 1}. {_clean(title)}\n   URL: {url}\n   {snippet}")
    if not results:
        return "Tidak ada hasil ditemukan."
    return "\n\n".join(results)


def tool_read_file(path, max_chars=CHAT_MAX_FILE_CHARS):
    """Baca isi file teks/kode untuk konteks agent (dengan batas ukuran)."""
    try:
        if not os.path.isfile(path):
            return None
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > 5:
            return f"(file terlalu besar: {size_mb:.1f} MB, dilewati)"
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read(max_chars)
        if len(data) == max_chars:
            data += "\n... (dipotong)"
        return data
    except Exception as e:
        return f"(gagal membaca file: {e})"


def tool_folder_tree(folder, max_files=CHAT_MAX_TREE_FILES, max_depth=5):
    """Susun struktur folder proyek (batas kedalaman & jumlah entri)."""
    lines = [f"{os.path.basename(folder) or folder}/"]
    count = 0
    for root, dirs, files in os.walk(folder):
        depth = os.path.relpath(root, folder).count(os.sep)
        if depth >= max_depth:
            dirs[:] = []
        dirs[:] = sorted(d for d in dirs
                         if d not in CHAT_SKIP_DIRS and not d.startswith("."))
        indent = "  " * (depth + 1)
        for d in dirs:
            lines.append(f"{indent}{d}/")
            count += 1
        for f in sorted(files):
            if count >= max_files:
                lines.append(f"{indent}... (batas {max_files} entri tercapai)")
                return "\n".join(lines)
            lines.append(f"{indent}{f}")
            count += 1
    return "\n".join(lines)


def tool_folder_files_preview(folder, max_files=CHAT_MAX_FOLDER_FILES,
                              max_chars=2000):
    """Baca cuplikan file kode/dok penting dari folder proyek untuk agent."""
    wanted_ext = set(SUPPORTED_CODE_EXT) | {".md", ".txt", ".json"}
    chunks = []
    n = 0
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs
                   if d not in CHAT_SKIP_DIRS and not d.startswith(".")]
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in wanted_ext:
                continue
            text = tool_read_file(os.path.join(root, fname), max_chars=max_chars)
            if not text or text.startswith("(file terlalu besar") or text.startswith("(gagal"):
                continue
            rel = os.path.relpath(os.path.join(root, fname), folder)
            chunks.append(f"--- {rel} ---\n{text}")
            n += 1
            if n >= max_files:
                break
        if n >= max_files:
            break
    if not chunks:
        return ""
    return "Cuplikan file penting dari folder:\n" + "\n\n".join(chunks)


def tool_system_info():
    """Informasi sistem (OS/CPU/RAM/GPU) untuk konteks agent."""
    lines = [f"OS: {sys.platform}",
             f"Python: {sys.version.split()[0]}",
             f"CPU cores: {os.cpu_count()}"]
    try:
        import psutil
        vm = psutil.virtual_memory()
        lines.append(f"RAM: {vm.used / (1024**3):.1f}/{vm.total / (1024**3):.1f} GB "
                     f"({vm.percent}%)")
    except ImportError:
        pass
    if torch is not None:
        lines.append(f"PyTorch CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            lines.append(f"GPU: {props.name} "
                         f"({props.total_memory / (1024**3):.1f} GB VRAM)")
    return "\n".join(lines)


# ================== UTILITY: QUEUE WRITER ==================
class QueueWriter:
    """Menulis output ke queue untuk GUI log."""
    def __init__(self, queue, original_stream=None):
        self.queue = queue
        self.original_stream = original_stream

    def write(self, text):
        self.queue.put(text)
        if self.original_stream:
            self.original_stream.write(text)

    def flush(self):
        pass

    def isatty(self):
        # Dipakai uvicorn/logging saat stdout dialihkan ke queue (server di thread)
        return False

    def fileno(self):
        # Fallback ke stream asli agar library yang butuh fd tetap bekerja
        if self.original_stream is not None:
            return self.original_stream.fileno()
        raise AttributeError("fileno tidak tersedia untuk QueueWriter")

# ================== SERVER THREAD ==================
class ServerThread(threading.Thread):
    """Thread untuk menjalankan server Flask dan MCP."""

    def __init__(self, model_source, model_path_or_id, port, device, log_queue, use_mcp=False, runtime_backend="pytorch", server_framework="fastapi", ext_backends=None):
        super().__init__(daemon=True)
        self.model_source = model_source
        self.model_path_or_id = model_path_or_id
        self.port = port
        self.device = device
        self.runtime_backend = runtime_backend
        self.server_framework = server_framework  # <-- BARU: Flask atau FastAPI
        self.log_queue = log_queue
        self.use_mcp = use_mcp
        self.ext_backends = ext_backends or []    # backend llama.cpp dari extensions/
        # Bertipe Any karena memuat objek dari banyak loader dinamis
        # (transformers, optimum.onnxruntime, llama_cpp, uvicorn, werkzeug, dll.)
        self.llm: Any = None
        self.pipeline: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.server: Any = None
        self.llm_server_proc: Any = None    # proses llama-server.exe (gguf-native)
        self.native_server_url: Any = None  # URL REST llama-server.exe (gguf-native)
        # Status thread server untuk monitoring GUI & CLI:
        #   "starting" -> sedang inisialisasi model
        #   "load-failed" -> model gagal dimuat (error tersimpan di self.error)
        #   "running"  -> model siap dan server API aktif
        #   "stopped"  -> dimatikan secara normal
        self.state = "starting"
        self.error = None

    # ---------- SERVER: FLASK ----------
    def _run_flask(self):
        from flask import Flask, request, jsonify
        app = Flask(__name__)

        @app.route('/', methods=['GET'])
        def index():
            return jsonify({"status": "running", "framework": "flask", "model": self.model_path_or_id})

        @app.route('/generate', methods=['POST'])
        def generate():
            data = request.get_json()
            prompt = data.get('prompt', '')
            response_text = self._generate(
                prompt,
                max_tokens=data.get('max_tokens', 256),
                temperature=data.get('temperature', 0.7),
                repetition_penalty=data.get('repetition_penalty', 1.1),
                top_p=data.get('top_p', 0.95)
            )
            return jsonify({"response": response_text})

        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        from werkzeug.serving import make_server
        self.server = make_server('0.0.0.0', self.port, app)
        self.log(f"Server berjalan di http://localhost:{self.port} (Framework: Flask)")
        self.server.serve_forever()

    # ---------- SERVER: FASTAPI ----------
    def _run_fastapi(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        import uvicorn

        app = FastAPI(title="Nusa Ai LLM Studio")

        @app.get('/')
        async def index():
            return {"status": "running", "framework": "fastapi", "model": self.model_path_or_id}

        @app.post('/generate')
        async def generate(request: Request):
            data = await request.json()
            prompt = data.get('prompt', '')
            try:
                response_text = self._generate(
                    prompt,
                    max_tokens=data.get('max_tokens', 256),
                    temperature=data.get('temperature', 0.7),
                    repetition_penalty=data.get('repetition_penalty', 1.1),
                    top_p=data.get('top_p', 0.95)
                )
                return {"response": response_text}
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})

        @app.get('/health')
        async def health():
            return {"status": "ok"}

        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="warning", timeout_keep_alive=15)
        self.server = uvicorn.Server(config)
        self.log(f"Server berjalan di http://localhost:{self.port} (Framework: FastAPI)")
        self.server.run()

    # ---------- MAIN RUN DISPATCHER ----------
    def run(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = QueueWriter(self.log_queue, original_stdout)
        sys.stderr = QueueWriter(self.log_queue, original_stderr)

        try:
            self.log(f"Memulai inisialisasi model {self.model_path_or_id}...")

            # Preset yang TIDAK dapat dimuat di server lokal (mis. Claude via API)
            # harus ditolak lebih awal dengan pesan yang jelas, bukan error HF.
            if self.model_path_or_id == "ANTHROPIC_API":
                raise ValueError(
                    "Preset 'Claude (API - tidak lokal)' tidak dapat dimuat di server "
                    "lokal. Pilih model lain (ID HuggingFace, path folder Transformers, "
                    "atau file .gguf).")

            # File .gguf SELALU diproses oleh loader GGUF (llama-server.exe native
            # dari extensions/backends atau llama-cpp-python), apapun sumber modelnya.
            if str(self.model_path_or_id).lower().endswith(".gguf"):
                self._load_gguf()
            else:
                self._load_transformers()

            if self.use_mcp:
                threading.Thread(target=self._run_mcp_server, daemon=True).start()

            # Model berhasil dimuat; status dipakai GUI/CLI untuk monitoring.
            self.state = "running"

            # Jalankan framework yang dipilih
            if self.server_framework == "fastapi":
                self._run_fastapi()
            else:
                self._run_flask()

        except Exception as e:
            self.state = "load-failed"
            self.error = str(e)
            self.log(f"ERROR Server: {e}")
            self.log(traceback.format_exc())
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if self.state not in ("running", "load-failed"):
                self.state = "stopped"

    def log(self, msg):
        self.log_queue.put(msg)

    # ---------- LOADER: GGUF / llama.cpp ----------
    def _load_gguf(self):
        """Load GGUF: native llama-server.exe (extensions/backends) atau llama-cpp-python."""
        if self.runtime_backend == "gguf-native":
            self._load_gguf_native()
            return
        self._load_llama_cpp()

    def _load_llama_cpp(self):
        """Load GGUF model menggunakan llama-cpp-python."""
        try:
            # FIX BUG (exe PyInstaller): llama-cpp-python melakukan
            # os.add_dll_directory(<pkg>/lib) saat import; pastikan folder
            # itu ADA di _MEIPASS agar tidak FileNotFoundError (WinError 3).
            if _is_frozen():
                try:
                    import llama_cpp as _lc_mod
                    _libdir = os.path.join(os.path.dirname(
                        os.path.abspath(_lc_mod.__file__)), "lib")
                    if not os.path.isdir(_libdir):
                        os.makedirs(_libdir, exist_ok=True)
                        self.log("[GGUF] Folder llama_cpp/lib tidak ada di bundle "
                                 "-> dibuat kosong (DLL harus di-bundle build).")
                except Exception:
                    pass
            from llama_cpp import Llama
            self.log("Memuat model GGUF...")
            if self.device != "cpu":
                self.log("Menggunakan GPU acceleration (Vulkan/CUDA) jika tersedia.")
                n_gpu_layers = -1  # offload semua layer ke GPU
            else:
                n_gpu_layers = 0
            self.llm = Llama(
                model_path=self.model_path_or_id,
                n_ctx=4096,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            self.log("Model GGUF berhasil dimuat.")
        except Exception as e:
            self.log(f"ERROR memuat GGUF: {e}")
            raise

    # ---------- LOADER: GGUF NATIVE (llama-server.exe dari extensions/backends) ----------
    def _load_gguf_native(self):
        """Jalankan llama-server.exe bawaan nusa_ai/extensions/backends
        (varian NVIDIA CUDA / AMD ROCm / Vulkan / CPU AVX2) dan gunakan
        REST API-nya sebagai engine inferensi GGUF tanpa llama-cpp-python.
        """
        backend = pick_llama_cpp_backend(self.device, backends=self.ext_backends,
                                         log_fn=self.log)
        if not backend:
            self.log("[Native] Tidak ada backend llama-server.exe di extensions/backends. "
                     "Fallback ke llama-cpp-python.")
            self._load_llama_cpp()
            return

        if requests is None:
            raise RuntimeError("Library 'requests' dibutuhkan untuk backend gguf-native.")

        internal_port = (self.port + 1) if self.port < 65535 else (self.port - 1)
        # port API native terpisah dari server utama (hindari overflow port)
        url = f"http://127.0.0.1:{internal_port}"
        n_gpu_layers = 0 if self.device == "cpu" else -1  # -1 = offload semua layer

        cmd = [
            backend["exe"],
            "-m", self.model_path_or_id,
            "--host", "127.0.0.1",
            "--port", str(internal_port),
            "-c", "4096",
            "-ngl", str(n_gpu_layers),
        ]
        env = prepare_backend_environment(backend)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        self.log(f"[Native] Memulai {os.path.basename(backend['exe'])} "
                 f"({backend['name']} v{backend['version']}, "
                 f"GPU: {backend['gpu_framework'] or 'CPU-only'}) di port {internal_port}...")
        if backend["vendor_dirs"]:
            self.log(f"[Native] DLL vendor: "
                     f"{', '.join(os.path.basename(v) for v in backend['vendor_dirs'])}")

        try:
            self.llm_server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                creationflags=creationflags,
                cwd=backend["dir"],
            )
        except Exception as e:
            self.log(f"ERROR menjalankan llama-server.exe: {e}")
            raise

        # Pump output llama-server.exe ke log GUI
        def _pump_output():
            try:
                for line in self.llm_server_proc.stdout:
                    if line.strip():
                        self.log(f"[llama-server] {line.strip()}")
            except Exception:
                pass

        threading.Thread(target=_pump_output, daemon=True).start()

        # Tunggu /health siap (proses load model GGUF bisa lama)
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.llm_server_proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server.exe keluar lebih awal (kode {self.llm_server_proc.returncode}).")
            try:
                r = requests.get(f"{url}/health", timeout=3)
                if r.status_code == 200:
                    self.native_server_url = url
                    self.log(f"[Native] llama-server siap di {url}. Model GGUF dimuat.")
                    return
            except requests.RequestException:
                pass
            time.sleep(1.0)
        raise RuntimeError("Timeout menunggu llama-server.exe siap (health check).")

    def stop(self):
        """Hentikan server API dan proses native llama-server.exe (jika ada)."""
        if self.server is not None:
            try:
                if self.server_framework == "fastapi":
                    self.server.should_exit = True
                else:
                    self.server.shutdown()
            except Exception as e:
                self.log(f"Peringatan saat mematikan server: {e}")
        proc = getattr(self, "llm_server_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.log("Server dihentikan.")

    # ---------- LOADER: PyTorch Transformer ----------
    def _load_pytorch_transformer(self):
        """Load model Transformers menggunakan PyTorch (CUDA/CPU)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch is None:
            self.log("ERROR: PyTorch tidak terinstall. Jalankan: pip install torch")
            raise ImportError("torch not installed")

        model_path = self.model_path_or_id

        # Jika path lokal berupa file .gguf, tangani terpisah
        if model_path.lower().endswith('.gguf'):
            self._load_gguf()
            return

        # Jika path lokal berupa file (bukan folder), cari folder config.json
        if os.path.isfile(model_path):
            folder = os.path.dirname(model_path)
            if os.path.exists(os.path.join(folder, "config.json")):
                model_path = folder
                self.log(f"Menggunakan folder model: {model_path}")
            else:
                self.log("ERROR: File model tidak didukung. Pilih folder model atau HF ID.")
                raise ValueError("Invalid model path")

        # Cek apakah path lokal atau HF ID
        if os.path.exists(model_path):
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                self.log("ERROR: config.json tidak ditemukan di folder model.")
                raise ValueError("Missing config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            architectures = config.get("architectures", [])
            if not any("ForCausalLM" in arch or "ForConditionalGeneration" in arch or "LMHead" in arch for arch in architectures):
                self.log("ERROR: Model ini bukan model generatif.")
                raise ValueError("Not a generative model")
            self.log(f"Memuat model lokal dari: {model_path}")
        else:
            self.log(f"Memuat model dari Hugging Face: {model_path}")
            
        # Tentukan device map dan dtype
        if self.device == "cuda":
            if torch.cuda.is_available():
                device_map = "cuda"
                dtype = torch.float16
            else:
                self.log("CUDA tidak tersedia, fallback ke CPU.")
                device_map = "cpu"
                dtype = torch.float32
        elif self.device == "directml":
            # jalur AMD/Intel: torch-directml (bisa untuk RX 500 series)
            try:
                import torch_directml  # type: ignore[import-untyped]
                dml_count = torch_directml.device_count() if hasattr(torch_directml, "device_count") else 1
                dml_names = []
                for i in range(dml_count):
                    try:
                        dml_names.append(torch_directml.device_name(i))
                    except Exception:
                        dml_names.append(f"GPU DirectML #{i}")
                dml_device = torch_directml.device()
                device_map = {"": dml_device}
                dtype = torch.float32  # Lebih stabil untuk RX 500 series (RX 560 XT)
                self.log(f"DirectML aktif! Terdeteksi {dml_count} GPU: {', '.join(dml_names)}")
            except Exception as e:
                self.log(f"ERROR DirectML PyTorch: {e}")
                self.log("Tip: untuk RX 560 XT pakai Runtime Backend=onnx (onnxruntime-directml) "
                         "atau install torch-directml yang cocok dengan versi torch.")
                device_map = "cpu"
                dtype = torch.float32
        elif self.device == "cpu":
            device_map = "cpu"
            dtype = torch.float32
        else:  # auto
            device_map = "auto"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.log(f"Device map: {device_map}, dtype: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            clean_up_tokenization_spaces=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # `torch_dtype` adalah parameter yang benar di seluruh versi transformers.
        # Ganti `dtype` (tidak dikenali oleh from_pretrained -> TypeError) dengan
        # `torch_dtype` agar model langsung termuat tanpa fallback.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

        from transformers import pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.log("Model PyTorch/Transformers berhasil dimuat.")

    # ---------- LOADER: ONNX Runtime ----------
    def _load_onnx(self):
        """Load model menggunakan ONNX Runtime (CUDA/DirectML/CPU)."""
        try:
            from optimum.onnxruntime import ORTModelForCausalLM  # type: ignore[import-untyped]
            from transformers import AutoTokenizer, pipeline
        except ImportError:
            self.log("Install optimum-onnx: pip install optimum-onnx onnxruntime-gpu")
            raise

        model_id = self.model_path_or_id
        
        if self.device == "cuda":
            provider = "CUDAExecutionProvider"
        elif self.device == "directml":
            # GPU AMD/Intel via DirectX (DmlExecutionProvider) - jalur paling stabil
            # untuk RX 560 XT
            if _dml_onnx_provider_available():
                provider = "DmlExecutionProvider"
                self.log("DirectML ONNX aktif: GPU AMD/Intel dipakai via DmlExecutionProvider.")
            else:
                provider = "CPUExecutionProvider"
                self.log("WARNING: DmlExecutionProvider belum tersedia. "
                         "Install onnxruntime-directml (Auto Install Dependencies) agar "
                         "RX 560 XT dipakai. Sementara memakai CPU.")
        elif self.device == "cpu":
            provider = "CPUExecutionProvider"
        else:  # auto
            if torch is not None and torch.cuda.is_available():
                provider = "CUDAExecutionProvider"
            else:
                provider = "CPUExecutionProvider"


        self.log(f"Memuat model ONNX dengan provider: {provider}")

        self.model = ORTModelForCausalLM.from_pretrained(
            model_id,
            export=True,               # Konversi otomatis dari PyTorch ke ONNX
            provider=provider,
            use_merged=True,           # Menghemat memori
            trust_remote_code=True,
            
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.log("Model ONNX berhasil dimuat.")

    # ---------- LOADER: OpenVINO ----------
    def _load_openvino(self):
        """Load model menggunakan OpenVINO (Intel CPU/GPU)."""
        try:
            from optimum.intel.openvino import OVModelForCausalLM  # type: ignore[import-untyped]
            from transformers import AutoTokenizer, pipeline
        except ImportError:
            self.log("Install optimum-intel: pip install optimum[intel] openvino")
            raise

        model_id = self.model_path_or_id

        # OpenVINO mendukung Intel GPU, bukan NVIDIA
        if self.device == "cpu":
            ov_device = "CPU"
        else:
            ov_device = "GPU"  # Gunakan Intel iGPU jika tersedia, fallback otomatis ke CPU

        self.log(f"Memuat model OpenVINO dengan device: {ov_device}")

        self.model = OVModelForCausalLM.from_pretrained(
            model_id,
            export=True,               # Konversi otomatis dari PyTorch ke OpenVINO IR
            device=ov_device,
            trust_remote_code=True,
            compile=False,             # Kompilasi dilakukan setelahnya
        )
        self.model.compile()           # Optimasi untuk inference

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.log("Model OpenVINO berhasil dimuat.")

    # ---------- DISPATCHER ----------
    def _load_transformers(self):
        """Load model Transformers berdasarkan runtime backend yang dipilih."""
        backend = self.runtime_backend
        try:
            if backend == "onnx":
                self._load_onnx()
            elif backend == "openvino":
                self._load_openvino()
            else:  # default: pytorch
                self._load_pytorch_transformer()
        except Exception as e:
            self.log(f"ERROR memuat Transformers: {e}")
            raise

    # ---------- GENERATE ----------
    
    def _generate(self, prompt, max_tokens=256, temperature=0.7, repetition_penalty=1.1, top_p=0.95):
        """Generate response menggunakan model apa pun."""
        if self.native_server_url is not None:  # GGUF native (llama-server.exe)
            if requests is None:
                return "Library 'requests' tidak tersedia untuk backend gguf-native."
            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "repeat_penalty": repetition_penalty,
                "top_p": top_p,
            }
            resp = requests.post(f"{self.native_server_url}/completion",
                                 json=payload, timeout=600)
            resp.raise_for_status()
            return resp.json().get("content", "")

        if self.llm is not None:  # GGUF / llama.cpp
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                repeat_penalty=repetition_penalty,
                top_p=top_p,
            )
            return output['choices'][0]['text']

        if self.pipeline is not None:  # PyTorch / ONNX / OpenVINO
            result = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                top_p=top_p,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            return result[0]['generated_text'].replace(prompt, "", 1).strip()

        return "Model belum dimuat."

    # ---------- MCP SERVER ----------
    def _run_mcp_server(self):
        """Jalankan MCP server (jika diminta)."""
        if not MCP_AVAILABLE or Server is None or Tool is None or TextContent is None:
            self.log("MCP tidak tersedia. Instal dengan: pip install mcp")
            return
        try:
            import asyncio
            from mcp.server.stdio import stdio_server

            # SDK mcp yang terpasang memakai nama parameter camelCase
            # (inputSchema) dan TIDAK menyediakan mcp.server.stdio.run();
            # pakai pola stdio_server() + Server.run() yang didukung SDK.
            # Stub SDK berbeda antar-versi (Pylance vs runtime), jadi baris
            # pemakaian diberi # type: ignore agar tidak jadi false positive.
            app = Server("local-model-server")  # type: ignore[call-arg, misc]

            @app.list_tools()  # type: ignore[attr-defined, misc]
            async def list_tools():
                return [
                    Tool(  # type: ignore[call-arg]
                        name="generate",
                        description="Generate text dari model lokal",
                                                inputSchema={  # type: ignore[reportCallIssue]  # kwargs SDK camelCase (mcp terpasang)
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"},
                                "max_tokens": {"type": "integer", "default": 256},
                                "temperature": {"type": "number", "default": 0.7}
                            },
                            "required": ["prompt"]
                        }
                    )
                ]

            @app.call_tool()  # type: ignore[attr-defined, misc]
            async def call_tool(name: str, arguments: dict):
                if name == "generate":
                    prompt = arguments["prompt"]
                    max_tokens = arguments.get("max_tokens", 256)
                    temperature = arguments.get("temperature", 0.7)
                    response = self._generate(prompt, max_tokens=max_tokens, temperature=temperature)
                    return [TextContent(type="text", text=response)]  # type: ignore[call-arg]
                raise ValueError(f"Unknown tool: {name}")

            self.log("MCP server berjalan...")

            async def _serve_stdio():
                async with stdio_server() as (read_stream, write_stream):
                    await app.run(  # type: ignore[attr-defined, misc]
                        read_stream,
                        write_stream,
                        app.create_initialization_options(),  # type: ignore[attr-defined, misc]
                    )

            asyncio.run(_serve_stdio())
        except Exception as e:
            self.log(f"ERROR MCP server: {e}")

# ================== TRAINING THREAD ==================
CODING_AGENT_SYSTEM_PROMPT = (
    "Anda adalah Nusa Ai Coding Agent, asisten pemrograman ahli yang membantu "
    "menulis, memperbaiki, dan menjelaskan kode dengan jelas serta lengkap."
)

class TrainingThread(threading.Thread):
    """Thread untuk fine-tuning model."""

    def __init__(self, base_model, dataset_path, output_dir, epochs, batch_size,
                 lr, use_lora, log_queue, max_length=512, system_prompt=None):
        super().__init__(daemon=True)
        self.base_model = base_model
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.use_lora = use_lora
        self.log_queue = log_queue
        self.max_length = max_length
        self.system_prompt = system_prompt or ""

    def log(self, msg):
        self.log_queue.put(msg)

    # ---------- PEMBACAAN DATASET MULTI FORMAT ----------
    def _load_samples(self):
        """Kembalikan daftar teks siap latih dari berbagai format dataset."""
        path = self.dataset_path
        texts = []
        ext = os.path.splitext(path)[1].lower()

        def fmt_alpaca(instr, inp, outp):
            prompt = f"{instr}\n\nInput:\n{inp}" if inp.strip() else instr
            return (
                f"### Sistem:\n{self.system_prompt}\n\n"
                f"### Instruksi:\n{prompt}\n\n### Respon:\n{outp}"
            )

        def from_messages(obj):
            parts = [f"### Sistem:\n{self.system_prompt}"] if self.system_prompt else []
            for msg in obj.get("messages", []):
                role = msg.get("role", "")
                content = msg.get("content", "")
                label = {"system": "Sistem", "user": "Pengguna", "assistant": "Asisten"}.get(role)
                if label:
                    parts.append(f"### {label}:\n{content}")
            return "\n\n".join(parts)

        def obj_to_text(obj):
            if isinstance(obj, dict) and "instruction" in obj:
                return fmt_alpaca(
                    str(obj["instruction"]),
                    str(obj.get("input", "") or ""),
                    str(obj.get("output", obj.get("response", "")) or ""),
                )
            if isinstance(obj, dict) and "messages" in obj:
                return from_messages(obj)
            if isinstance(obj, dict) and ("text" in obj or "content" in obj):
                return str(obj.get("text") or obj.get("content"))
            return None

        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    t = obj_to_text(obj)
                    texts.append(t if t is not None else json.dumps(obj, ensure_ascii=False))
        elif ext == ".json":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                try:
                    obj = json.load(f)
                except json.JSONDecodeError:
                    raise ValueError("File JSON tidak valid.")
            items = obj if isinstance(obj, list) else [obj]
            for item in items:
                t = obj_to_text(item) if isinstance(item, dict) else None
                if t is not None:
                    texts.append(t)
        else:
            # .txt / lainnya: satu contoh per baris non-kosong
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                texts = [line.strip() for line in f if line.strip()]

        return [t for t in texts if isinstance(t, str) and t.strip()]

    def run(self):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer, Trainer,
                TrainingArguments, DataCollatorForLanguageModeling,
            )
            from datasets import Dataset  # type: ignore[import-untyped]

            self.log("Memulai training...")
            self.log(f"Base model: {self.base_model}")
            self.log(f"Dataset: {self.dataset_path}")
            if self.system_prompt:
                self.log(f"Sistem peran: {self.system_prompt[:80]}...")

            tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            texts = self._load_samples()
            if not texts:
                self.log("ERROR: Dataset kosong atau tidak ada sampel valid.")
                return

            self.log(f"Jumlah sampel: {len(texts)}")

            def tokenize_function(examples):
                return tokenizer(examples["text"], truncation=True, max_length=self.max_length)

            dataset = Dataset.from_dict({"text": texts})
            dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            # Pilih device & dtype otomatis
            if torch.cuda.is_available():
                device_map = "cuda"
                dtype = torch.float16
                self.log(f"Training di GPU: {torch.cuda.get_device_name(0)}")
            else:
                device_map = None
                dtype = torch.float32
                self.log("CUDA tidak tersedia, training di CPU (akan lebih lambat).")

            model_kwargs: "dict[str, Any]" = {"trust_remote_code": True}
            if device_map:
                model_kwargs["device_map"] = device_map

            model = AutoModelForCausalLM.from_pretrained(
                self.base_model, torch_dtype=dtype, **model_kwargs)

            if self.use_lora:
                try:
                    from peft import LoraConfig, get_peft_model, TaskType
                    lora_config = LoraConfig(
                        r=8,
                        lora_alpha=32,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type=TaskType.CAUSAL_LM
                    )
                    model = get_peft_model(model, lora_config)
                    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    self.log(f"LoRA diaktifkan ({trainable:,} parameter trainable).")
                except ImportError:
                    self.log("LoRA tidak tersedia, lanjut tanpa LoRA.")

            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.batch_size,
                learning_rate=self.lr,
                save_strategy="epoch",
                logging_steps=10,
                logging_dir=os.path.join(self.output_dir, "logs"),
                report_to=[],
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )

            trainer.train()
            trainer.save_model(self.output_dir)
            tokenizer.save_pretrained(self.output_dir)

            # Simpan metadata agar model mudah dikenali saat disinkronkan ke models/
            meta_path = os.path.join(self.output_dir, "training_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump({
                    "base_model": self.base_model,
                    "dataset": self.dataset_path,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.lr,
                    "use_lora": bool(self.use_lora),
                    "system_prompt": self.system_prompt,
                    "purpose": ("coding-agent" if "coding agent" in (self.system_prompt or "").lower()
                                else "general"),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, mf, indent=2, ensure_ascii=False)

            # Sinkronkan hasil training ke registry preset lokal
            save_model_sync_config(self.base_model)

            self.log(f"Training selesai. Model disimpan di: {self.output_dir}")
            self.log("Model siap dipakai: tab Server -> 'Sync Model Lokal' -> pilih preset [Lokal].")

        except Exception as e:
            self.log(f"ERROR Training: {e}")
            self.log(traceback.format_exc())

# ================== GUI APPLICATION ==================
class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nusa Ai LLM Studio")
        self.root.geometry("900x700")

        # Status
        self.server_running = False
        self.server_thread = None
        self.training_thread = None

        # Variabel runtime settings
        self.max_tokens_var = IntVar(value=256)
        self.temperature_var = DoubleVar(value=0.7)
        self.repetition_penalty_var = DoubleVar(value=1.1)
        self.top_p_var = DoubleVar(value=0.95)
        self.request_timeout_var = IntVar(value=300)

        # Variabel server
        self.server_port = StringVar(value="8000")
        self.server_model_path = StringVar(value="")
        self.server_model_id = StringVar(value="Qwen/Qwen2.5-1.5B-Instruct")
        self.model_source_var = StringVar(value="local")
        self.device_var = StringVar(value="auto")
        self.runtime_backend_var = StringVar(value="pytorch")
        self.model_preset_var = StringVar(value="Qwen2.5-1.5B-Instruct")
        
        self.server_framework_var = StringVar(value="fastapi")

        # Log queue
        self.log_queue = queue.Queue()
        self.is_logging = False

        # State chat agent (riwayat, lampiran file/folder, status generasi)
        self.chat_messages = []
        self.chat_attachments = []
        self._chat_busy = False
        self._chat_cancelled = False

        # SINKRONISASI MODEL LOKAL: gabungkan isi folder models/ ke daftar preset
        MODEL_PRESETS.update(CODING_AGENT_PRESETS)
        sync_model_presets(log_fn=self.log)

        # BONSAI AI SEBAGAI AGENT: daftarkan preset Bonsai (GGUF lokal) jika ada
        bonsai_path = register_bonsai_agent_preset()
        if bonsai_path:
            self.log(f"[Agent] Bonsai AI terdaftar: {BONSAI_PRESET_NAME} -> {bonsai_path}")
        else:
            self.log("[Agent] GGUF Bonsai AI tidak ditemukan di lokasi kandidat.")

        # SINKRONISASI EXTENSIONS: backends llama.cpp (llama-server.exe),
        # frameworks (harmony, lmlink-connector), dan plugins (lmstudio/mcp)
        self.ext_backends = discover_llama_cpp_backends(log_fn=self.log)
        self.ext_frameworks = discover_extension_frameworks(log_fn=self.log)
        self.ext_plugins = discover_extension_plugins(log_fn=self.log)
        self.log(f"[Ext] Sinkronisasi extensions selesai: {len(self.ext_backends)} backend, "
                 f"{len(self.ext_frameworks)} framework, {len(self.ext_plugins)} plugin.")

        # SINKRONISASI MODUL NUSAAI_CODEASSIST (ekstensi editor -> SATU server)
        self.codeassist_info = discover_codeassist_extension(log_fn=self.log)

        # Deteksi GPU AMD/Intel (DirectML) untuk men-support kartu seperti RX 560 XT
        dml_devices = detect_directml_devices()
        if dml_devices:
            self.log("[DirectML] GPU AMD/Intel terdeteksi:")
            for _d in dml_devices:
                self.log(f"    - {_d}")
        else:
            self.log("[DirectML] Tidak ada akselerasi DirectML. Untuk GPU AMD (mis. "
                     "RX 560 XT) pilih Device=directml lalu jalankan 'Auto Install Dependencies'.")

        # Inisialisasi UI
        self._create_widgets()
        self._poll_log_queue()

    def _create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=True)

        # Existing Tabs
        self._create_server_tab()
        self._create_chat_tab()
        self._create_training_tab()
        self._create_dataset_tab()
        self._create_settings_tab()
        
        # New Feature Tabs
        self._create_agents_tab()
        self._create_analytics_tab()
        self._create_communication_tab()
        self._create_integrations_tab()
        self._create_advanced_ai_tab()
        self._create_security_tab()
        self._create_skills_tab()
        self._create_monitoring_tab()

        # Log Tab (kept at the end)
        self._create_log_tab()
        
    def auto_install_dependencies(self):
        """Trigger thread instalasi agar GUI tidak freeze."""
        if messagebox.askyesno("Konfirmasi Instalasi", "Ini akan mengunduh dan menginstal pustaka Python yang dibutuhkan. Lanjutkan?"):
            threading.Thread(target=self._install_dependencies_thread, daemon=True).start()

    def _install_dependencies_thread(self):
        """Menjalankan proses pip install dengan aturan pengecekan module."""
        self.install_btn.config(state=DISABLED)
        self.log("Memeriksa Aturan Instalasi (Installation Rules)...")

        # Rule 1: Daftar dependensi dasar (Format -> "nama_di_pip": "nama_module_python")
        # Rule 1: Daftar dependensi dasar (Ganti Flask dengan FastAPI & Uvicorn)
        required_packages = {
            "torch": "torch",
            "transformers": "transformers",
            "fastapi": "fastapi",           # <-- BARU
            "uvicorn": "uvicorn",           # <-- BARU
            "pydantic": "pydantic",         # <-- BARU
            "requests": "requests",
            "datasets": "datasets",
            "peft": "peft",
            "accelerate": "accelerate"
        }

        # Aturan 2: Tambahkan dependensi spesifik berdasarkan backend & device yang dipilih
        runtime = self.runtime_backend_var.get()
        device_sel = self.device_var.get()
        if runtime == "onnx":
            if device_sel == "directml":
                # Saat ONNX + DirectML, jangan pakai extra [onnxruntime] (agar bisa
                # install onnxruntime-directml tanpa bentrok resolver pip dengan onnxruntime).
                required_packages["optimum"] = "optimum"
            else:
                required_packages["optimum[onnxruntime]"] = "optimum"
        elif runtime == "openvino":
            required_packages["optimum[intel]"] = "optimum"
            required_packages["openvino"] = "openvino"
        elif runtime == "gguf-native":
            # Backend native memakai llama-server.exe bawaan nusa_ai/extensions/backends
            self.log("[*] Rule pass: runtime 'gguf-native' memakai llama-server.exe "
                     "bawaan extensions/backends (tidak butuh paket pip tambahan).")
        elif runtime == "gguf-vulkan":
            required_packages["llama-cpp-python"] = "llama_cpp"

        # Rule 2b: Akselerasi untuk GPU AMD/Intel jika Device = directml
        dml_extra = []
        device_sel = self.device_var.get()
        if device_sel == "directml":
            self.log("[DirectML] Device=directml dipilih untuk GPU AMD/Intel (mis. RX 560 XT).")
            if runtime == "onnx":
                # Jalur paling stabil untuk RX 560 XT: onnxruntime-directml
                # (menggantikan onnxruntime/onnxruntime-gpu karena berbagi nama modul)
                if _dml_onnx_provider_available():
                    self.log("[*] Rule pass: DmlExecutionProvider sudah tersedia, melewati.")
                else:
                    self.log("[*] Rule: install onnxruntime-directml untuk DmlExecutionProvider. "
                             "Catatan: paket ini MENGANTI onnxruntime/onnxruntime-gpu, "
                             "agar DML (AMD/Intel) bisa dipakai.")
                    dml_extra.append("onnxruntime-directml")
            elif runtime == "pytorch":
                # Pytorch: torch-directml; catatan versi torch 2.6+ belum didukung
                # torch-directml, jadi disarankan backend onnx untuk GPU AMD.
                if importlib.util.find_spec("torch_directml") is None:
                    dml_extra.append("torch-directml")
                else:
                    self.log("[*] Rule pass: torch-directml sudah terinstal, melewati.")

        # Rule 3: Tambahkan MCP jika diaktifkan
        if self.use_mcp_var.get():
            required_packages["mcp"] = "mcp"

        # Rule 4: Filter hanya paket yang BELUM terinstal
        packages_to_install = []
        for pip_name, mod_name in required_packages.items():
            # Cek apakah modul sudah ada di sistem
            if importlib.util.find_spec(mod_name) is None:
                packages_to_install.append(pip_name)
            else:
                self.log(f"[*] Rule pass: '{pip_name}' sudah terinstal, melewati.")
        # Gabungkan paket DirectML (hasil Rule 2b)
        if dml_extra:
            packages_to_install.extend([p for p in dml_extra])
            self.log(f"[DirectML] Paket DirectML tambahan: {', '.join(dml_extra)}")

        # Jika array kosong, berarti semua sudah lengkap
        if not packages_to_install:
            self.log("Semua dependensi sudah lengkap. Menghentikan proses instalasi.")
            messagebox.showinfo("Info", "Semua dependensi sudah terinstal di sistem Anda!")
            self.install_btn.config(state=NORMAL)
            return

        self.log(f"Paket yang akan diinstal: {', '.join(packages_to_install)}")
        self.log("Memulai pengunduhan. Mohon tunggu...")

        try:
            cmd = [sys.executable, "-m", "pip", "install"] + packages_to_install
            
            creationflags = 0
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NO_WINDOW

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creationflags
            )

            if process.stdout is not None:
                for line in process.stdout:
                    self.log(line.strip())

            process.wait()

            if process.returncode == 0:
                self.log("Instalasi dependensi selesai dengan sukses!")
                messagebox.showinfo("Sukses", "Instalasi dependensi berhasil! Silakan restart aplikasi.")
            else:
                self.log("Peringatan: Terdapat error saat instalasi dependensi.")
                messagebox.showerror("Error", "Gagal menginstal beberapa dependensi. Silakan cek tab Log.")

        except Exception as e:
            self.log(f"ERROR instalasi: {e}")
            messagebox.showerror("Error", f"Terjadi kesalahan kritis: {e}")
        finally:
            self.install_btn.config(state=NORMAL)
        
    # ------------------ AGENTS TAB ------------------
    def _create_agents_tab(self):
        agents_tab = ttk.Frame(self.notebook)
        self.notebook.add(agents_tab, text="Agents")

        frame = ttk.LabelFrame(agents_tab, text="Agent Configuration")
        frame.pack(fill=X, padx=10, pady=10)

        ttk.Label(frame, text="Agent Name:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        self.agent_name_var = StringVar()
        ttk.Entry(frame, textvariable=self.agent_name_var, width=30).grid(row=0, column=1, padx=5, pady=5)

        # Model Agent (termasuk Bonsai AI lokal sebagai agent)
        ttk.Label(frame, text="Model Agent:").grid(row=1, column=0, padx=5, pady=5, sticky=W)
        self.agent_model_var = StringVar(value=BONSAI_PRESET_NAME if BONSAI_PRESET_NAME in MODEL_PRESETS else "")
        agent_models = sorted(
            {BONSAI_PRESET_NAME} | set(CODING_AGENT_PRESETS.keys()),
        )
        agent_models = [m for m in agent_models if m in MODEL_PRESETS]
        self.agent_model_combo = ttk.Combobox(frame, textvariable=self.agent_model_var,
                                              values=agent_models, width=38, state="readonly")
        self.agent_model_combo.grid(row=1, column=1, padx=5, pady=5, sticky=W)

        ttk.Button(frame, text="Pakai Model Ini di Server",
                   command=self.use_agent_model).grid(row=1, column=2, padx=5)

        ttk.Label(frame, text="Preferred Tools:").grid(row=2, column=0, padx=5, pady=5, sticky=W)
        
        self.tool_search_var = IntVar()
        self.tool_calc_var = IntVar()
        self.tool_file_var = IntVar()
        
        tools_frame = ttk.Frame(frame)
        tools_frame.grid(row=2, column=1, sticky=W)
        ttk.Checkbutton(tools_frame, text="Web Search", variable=self.tool_search_var).pack(side=LEFT, padx=5)
        ttk.Checkbutton(tools_frame, text="Calculator", variable=self.tool_calc_var).pack(side=LEFT, padx=5)
        ttk.Checkbutton(tools_frame, text="File System", variable=self.tool_file_var).pack(side=LEFT, padx=5)

        ttk.Button(frame, text="Create Agent", command=lambda: self.log(f"Agent '{self.agent_name_var.get()}' created (model: {self.agent_model_var.get()}).")).grid(row=3, column=0, columnspan=2, pady=10)

    def use_agent_model(self):
        """Terapkan model agent yang dipilih (mis. Bonsai AI) ke konfigurasi server."""
        preset = self.agent_model_var.get()
        if preset not in MODEL_PRESETS:
            messagebox.showerror("Error", "Preset model agent tidak dikenal.")
            return
        self.model_preset_var.set(preset)
        self._update_model_from_preset()
        self.notebook.select(0)  # pindah fokus ke tab Server

    # ------------------ ANALYTICS TAB ------------------
    def _create_analytics_tab(self):
        analytics_tab = ttk.Frame(self.notebook)
        self.notebook.add(analytics_tab, text="Analytics")

        frame = ttk.LabelFrame(analytics_tab, text="Session Tracking & Recommendations")
        frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.analytics_text = scrolledtext.ScrolledText(frame, state='disabled', height=10)
        self.analytics_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Mock Data Injection
        self.analytics_text.config(state='normal')
        self.analytics_text.insert(END, "Session ID: 9823-ABCD\nActive Time: 45 mins\nUser Intent: Code Generation\n")
        self.analytics_text.insert(END, "Recommendation: Consider enabling 'Logic' skills for better code structuring.\n")
        self.analytics_text.config(state='disabled')

        ttk.Button(frame, text="Refresh Analytics", command=lambda: self.log("Analytics refreshed.")).pack(pady=5)

    # ------------------ COMMUNICATION TAB ------------------
    def _create_communication_tab(self):
        comm_tab = ttk.Frame(self.notebook)
        self.notebook.add(comm_tab, text="Communication")

        frame = ttk.LabelFrame(comm_tab, text="Omnichannel & NLP Integration")
        frame.pack(fill=X, padx=10, pady=10)

        ttk.Label(frame, text="Active Channels:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        
        self.chan_web_var = IntVar(value=1)
        self.chan_msg_var = IntVar()
        self.chan_voice_var = IntVar()
        
        ttk.Checkbutton(frame, text="Web Chat", variable=self.chan_web_var).grid(row=0, column=1, sticky=W)
        ttk.Checkbutton(frame, text="Messaging Apps (WhatsApp/Telegram)", variable=self.chan_msg_var).grid(row=1, column=1, sticky=W)
        ttk.Checkbutton(frame, text="Voice Calls", variable=self.chan_voice_var).grid(row=2, column=1, sticky=W)

        ttk.Label(frame, text="NLP Engine:").grid(row=3, column=0, padx=5, pady=5, sticky=W)
        ttk.Combobox(frame, values=["Standard", "Advanced Intent Recognition", "Contextual Semantic"], state="readonly").grid(row=3, column=1, padx=5, pady=5, sticky=W)

    # ------------------ INTEGRATIONS TAB ------------------
    def _create_integrations_tab(self):
        integ_tab = ttk.Frame(self.notebook)
        self.notebook.add(integ_tab, text="Integrations")

        frame = ttk.LabelFrame(integ_tab, text="External Services (APIs)")
        frame.pack(fill=X, padx=10, pady=10)

        ttk.Label(frame, text="Calendar API Key:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        ttk.Entry(frame, width=40, show="*").grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Weather API Key:").grid(row=1, column=0, padx=5, pady=5, sticky=W)
        ttk.Entry(frame, width=40, show="*").grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame, text="News Feed URL:").grid(row=2, column=0, padx=5, pady=5, sticky=W)
        ttk.Entry(frame, width=40).grid(row=2, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Test Connections", command=lambda: self.log("Testing API connections...")).grid(row=3, column=0, columnspan=2, pady=10)

    # ------------------ ADVANCED AI TAB ------------------
    def _create_advanced_ai_tab(self):
        ai_tab = ttk.Frame(self.notebook)
        self.notebook.add(ai_tab, text="Advanced AI")

        frame = ttk.LabelFrame(ai_tab, text="Techniques & Architectures")
        frame.pack(fill=X, padx=10, pady=10)

        self.ai_ml_var = IntVar(value=1)
        self.ai_dl_var = IntVar(value=1)
        self.ai_rule_var = IntVar()
        self.ai_kg_var = IntVar()

        ttk.Checkbutton(frame, text="Machine Learning Algorithms", variable=self.ai_ml_var).grid(row=0, column=0, padx=5, pady=5, sticky=W)
        ttk.Checkbutton(frame, text="Deep Learning Models", variable=self.ai_dl_var).grid(row=1, column=0, padx=5, pady=5, sticky=W)
        ttk.Checkbutton(frame, text="Rule-Based Systems", variable=self.ai_rule_var).grid(row=2, column=0, padx=5, pady=5, sticky=W)
        ttk.Checkbutton(frame, text="Knowledge Graphs", variable=self.ai_kg_var).grid(row=3, column=0, padx=5, pady=5, sticky=W)

    # ------------------ SECURITY TAB ------------------
    def _create_security_tab(self):
        sec_tab = ttk.Frame(self.notebook)
        self.notebook.add(sec_tab, text="Security")

        frame = ttk.LabelFrame(sec_tab, text="Data Privacy & Compliance")
        frame.pack(fill=X, padx=10, pady=10)

        self.sec_enc_var = IntVar(value=1)
        self.sec_access_var = IntVar(value=1)

        ttk.Checkbutton(frame, text="Enable AES-256 Data Encryption", variable=self.sec_enc_var).grid(row=0, column=0, padx=5, pady=5, sticky=W)
        ttk.Checkbutton(frame, text="Strict Role-Based Access Controls (RBAC)", variable=self.sec_access_var).grid(row=1, column=0, padx=5, pady=5, sticky=W)
        
        ttk.Label(frame, text="Compliance Standard:").grid(row=2, column=0, padx=5, pady=5, sticky=W)
        ttk.Combobox(frame, values=["None", "GDPR", "HIPAA", "SOC2"], state="readonly").grid(row=2, column=1, padx=5, pady=5, sticky=W)

    # ------------------ SKILLS TAB ------------------
    def _create_skills_tab(self):
        skills_tab = ttk.Frame(self.notebook)
        self.notebook.add(skills_tab, text="Skills")

        frame = ttk.LabelFrame(skills_tab, text="Personality & Cognitive Skill Sets")
        frame.pack(fill=X, padx=10, pady=10)

        skills = ["Creativity", "Empathy", "Logic", "Critical Thinking", "Curiosity", "Adaptability", "Emotional Intelligence"]
        self.skill_vars = {}

        for i, skill in enumerate(skills):
            ttk.Label(frame, text=f"{skill}:").grid(row=i, column=0, padx=5, pady=5, sticky=W)
            var = DoubleVar(value=0.5)
            self.skill_vars[skill] = var
            ttk.Scale(frame, from_=0.0, to=1.0, variable=var, orient=HORIZONTAL, length=150).grid(row=i, column=1, padx=5)

    # ------------------ MONITORING TAB ------------------
    def _create_monitoring_tab(self):
        mon_tab = ttk.Frame(self.notebook)
        self.notebook.add(mon_tab, text="Monitoring")

        frame = ttk.LabelFrame(mon_tab, text="System Performance & Feedback")
        frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.mon_text = scrolledtext.ScrolledText(frame, state='disabled', height=10)
        self.mon_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # Mock Monitoring Data
        self.mon_text.config(state='normal')
        self.mon_text.insert(END, "Resource Usage:\n- CPU: 24%\n- RAM: 4.2 GB\n- GPU VRAM: 6.8 GB\n\n")
        self.mon_text.insert(END, "Goal Progress: 85% (Model Optimization)\nUser Feedback Score: 4.8/5.0\n")
        self.mon_text.config(state='disabled')

        ttk.Button(frame, text="Run Diagnostics", command=lambda: self.log("Diagnostics initiated...")).pack(pady=5)

    # ------------------ SERVER TAB ------------------
    def _load_logo_tk(self, max_dim=220):
        """Muat logo resmi sebagai PhotoImage berukuran kecil untuk banner GUI.
        Referensi disimpan di self._logo_ref supaya tidak di-garbage-collected."""
        try:
            from PIL import Image, ImageTk
            for cand in (LOGO_PNG, LOGO_GIF, LOGO_ICO):
                if not os.path.isfile(cand):
                    continue
                try:
                    img = Image.open(cand)
                    if getattr(img, "mode", None) not in ("RGBA", "RGB", "L", "P"):
                        img = img.convert("RGBA")
                    img.thumbnail((max_dim, max_dim))
                    self._logo_ref = ImageTk.PhotoImage(img)
                    return self._logo_ref
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def _create_server_tab(self):
        server_tab = ttk.Frame(self.notebook)
        self.notebook.add(server_tab, text="Server")

        # Banner logo resmi di header tab Server
        logo_img = self._load_logo_tk(max_dim=190)
        if logo_img is not None:
            header = ttk.Frame(server_tab)
            header.pack(fill=X, padx=10, pady=(10, 2))
            ttk.Label(header, image=logo_img).pack(side=LEFT, padx=(0, 10))
            ttk.Label(header, text=APP_NAME, font=("Segoe UI", 16, "bold")).pack(anchor="w")
            ttk.Label(header, text=f"v{APP_VERSION} - LLM Server, Chat, Training & Dataset Lokal",
                      foreground="gray").pack(anchor="w", padx=(2, 0))

        frame = ttk.LabelFrame(server_tab, text="Konfigurasi Model")
        frame.pack(fill=X, padx=10, pady=10)

        # Model source
        ttk.Label(frame, text="Sumber Model:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        source_combo = ttk.Combobox(frame, textvariable=self.model_source_var, values=["local", "huggingface"], state="readonly")
        source_combo.grid(row=0, column=1, padx=5, pady=5, sticky=W)
        source_combo.bind("<<ComboboxSelected>>", lambda e: self._update_model_source_ui())

        # Model preset (HF)
        ttk.Label(frame, text="Preset Model:").grid(row=1, column=0, padx=5, pady=5, sticky=W)
        preset_combo = ttk.Combobox(frame, textvariable=self.model_preset_var, values=list(MODEL_PRESETS.keys()), state="readonly")
        self.preset_combo = preset_combo
        preset_combo.grid(row=1, column=1, padx=5, pady=5, sticky=W)
        preset_combo.bind("<<ComboboxSelected>>", lambda e: self._update_model_from_preset())

        # Model ID / Path
        self.model_id_label = ttk.Label(frame, text="Model ID (HF):")
        self.model_id_label.grid(row=2, column=0, padx=5, pady=5, sticky=W)
        self.model_id_entry = ttk.Entry(frame, textvariable=self.server_model_id, width=50)
        self.model_id_entry.grid(row=2, column=1, padx=5, pady=5)

        self.model_path_label = ttk.Label(frame, text="Model Path (Lokal):")
        self.model_path_label.grid(row=3, column=0, padx=5, pady=5, sticky=W)
        self.model_path_entry = ttk.Entry(frame, textvariable=self.server_model_path, width=50)
        self.model_path_entry.grid(row=3, column=1, padx=5, pady=5)
        browse_btn = ttk.Button(frame, text="Browse", command=self.browse_server_model)
        browse_btn.grid(row=3, column=2, padx=5, pady=5)

        # Port
        ttk.Label(frame, text="Port:").grid(row=4, column=0, padx=5, pady=5, sticky=W)
        ttk.Entry(frame, textvariable=self.server_port, width=10).grid(row=4, column=1, padx=5, pady=5, sticky=W)

        # Device
        ttk.Label(frame, text="Device:").grid(row=5, column=0, padx=5, pady=5, sticky=W)
        device_combo = ttk.Combobox(frame, textvariable=self.device_var,
                                    values=["auto", "cuda", "directml", "vulkan", "cpu"],
                                    state="readonly")
        device_combo.grid(row=5, column=1, padx=5, pady=5, sticky=W)

        # Runtime Backend (TAMBAHAN BARU)
        ttk.Label(frame, text="Runtime Backend:").grid(row=6, column=0, padx=5, pady=5, sticky=W)
        runtime_combo = ttk.Combobox(frame, textvariable=self.runtime_backend_var,
                                     values=["pytorch", "onnx", "openvino",
                                             "gguf-native", "gguf-vulkan"],
                                     state="readonly")
        runtime_combo.grid(row=6, column=1, padx=5, pady=5, sticky=W)
        
        # Framework API Selection
        ttk.Label(frame, text="Server Framework:").grid(row=7, column=0, padx=5, pady=5, sticky=W)
        framework_combo = ttk.Combobox(frame, textvariable=self.server_framework_var, values=["fastapi", "flask"], state="readonly")
        framework_combo.grid(row=7, column=1, padx=5, pady=5, sticky=W)

        # MCP (geser ke row 8)
        self.use_mcp_var = IntVar(value=1 if MCP_AVAILABLE else 0)
        ttk.Checkbutton(frame, text="Aktifkan MCP (jika tersedia)", variable=self.use_mcp_var).grid(row=8, column=0, columnspan=2, padx=5, pady=5, sticky=W)

        # Info backend extensions (llama-server.exe varian CUDA/ROCm/Vulkan/CPU)
        if self.ext_backends:
            ext_summary = ", ".join(
                f"{b['gpu_framework'] or 'CPU'} v{b['version']}" for b in self.ext_backends)
            ttk.Label(frame, foreground="green",
                      text=(f"Backends llama-server terdeteksi: {len(self.ext_backends)} "
                            f"({ext_summary})")).grid(row=9, column=0, columnspan=3,
                                                      padx=5, pady=5, sticky=W)


        # Tombol start/stop dan install
        btn_frame = ttk.Frame(server_tab)
        btn_frame.pack(pady=10)
        self.start_btn = ttk.Button(btn_frame, text="Start Server", command=self.start_server)
        self.start_btn.pack(side=LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="Stop Server", command=self.stop_server, state=DISABLED)
        self.stop_btn.pack(side=LEFT, padx=5)
        
        # --- SYNC & INSTALL BUTTONS ---
        self.install_btn = ttk.Button(btn_frame, text="Auto Install Dependencies", command=self.auto_install_dependencies)
        self.install_btn.pack(side=LEFT, padx=5)
        self.sync_btn = ttk.Button(btn_frame, text="Sync Model Lokal", command=self.sync_local_models_ui)
        self.sync_btn.pack(side=LEFT, padx=5)
        self.codeassist_btn = ttk.Button(btn_frame, text="Sync CodeAssist",
                                         command=self.sync_codeassist_ui)
        self.codeassist_btn.pack(side=LEFT, padx=5)

        # Info
        info_text = """
        Pilih sumber model:
        - local: folder model Transformers (berisi config.json) atau file .gguf
        - huggingface: ID model dari Hugging Face (contoh: Qwen/Qwen2.5-1.5B-Instruct)

        Runtime Backend:
        - pytorch: Menggunakan PyTorch (CUDA untuk NVIDIA / torch-directml untuk AMD)
        - onnx: Menggunakan ONNX Runtime (CUDA/DirectML/CPU) - lebih ringan
          -> JALUR TERBAIK UNTUK AMD RX 560 XT: pilih Device=directml + backend ini,
             lalu 'Auto Install Dependencies' (akan masuk onnxruntime-directml)
        - openvino: Menggunakan OpenVINO (Intel CPU/iGPU) - optimal untuk Intel
        - gguf-native: File .gguf dijalankan llama-server.exe bawaan
          nusa_ai/extensions/backends (varian NVIDIA CUDA / AMD ROCm / Vulkan /
          CPU AVX2 dipilih otomatis sesuai Device, versi tertinggi)
        - gguf-vulkan: Untuk file .gguf via llama-cpp-python (Vulkan = jalur AMD alternatif)

        Device:
        - directml: GPU AMD/Intel (RX 560 XT, RX 500 series, iGPU) via DirectX
        - vulkan:  Untuk GGUF/llama.cpp lewat Vulkan (AMD)

        Preset model mencakup Qwen, Llama, Mistral, Phi, Gemma, DeepSeek,
        Coding Agent (Qwen2.5-Coder/CodeGemma), model lokal dari folder models/
        (termasuk nusa_ai/extensions/models) dengan awalan [Lokal], dan Claude (API).
        """

        ttk.Label(server_tab, text=info_text, justify=LEFT).pack(padx=10, pady=5)

        self._update_model_source_ui()

    def _update_model_source_ui(self):
        if self.model_source_var.get() == "huggingface":
            self.model_id_label.grid()
            self.model_id_entry.grid()
            self.model_path_label.grid_remove()
            self.model_path_entry.grid_remove()
        else:
            self.model_id_label.grid_remove()
            self.model_id_entry.grid_remove()
            self.model_path_label.grid()
            self.model_path_entry.grid()

    def _update_model_from_preset(self):
        preset = self.model_preset_var.get()
        if preset in MODEL_PRESETS:
            model_value = MODEL_PRESETS[preset]
            self.server_model_id.set(model_value)
            # Preset yang nilainya path lokal yang benar-benar ada (mis. Bonsai GGUF)
            # otomatis memakai sumber 'local', bukan 'huggingface'.
            if os.path.exists(model_value) and (os.path.isfile(model_value) or os.path.isdir(model_value)):
                self.model_source_var.set("local")
                self.server_model_path.set(model_value)
                self.log(f"Model lokal dipilih: {preset} -> {model_value}")
                if model_value.lower().endswith(".gguf"):
                    # Utamakan backend native llama-server.exe dari extensions/backends
                    self.runtime_backend_var.set(
                        "gguf-native" if self.ext_backends else "gguf-vulkan")
                self._update_model_source_ui()
            elif preset == "Claude (API - tidak lokal)":
                messagebox.showwarning("Info", "Claude tidak dapat dimuat secara lokal. Gunakan API Anthropic secara terpisah.")
            elif preset.startswith(LOCAL_MODEL_PREFIX):
                # Model lokal hasil sinkronisasi folder models/
                self.model_source_var.set("local")
                self.server_model_path.set(MODEL_PRESETS[preset])
                self.log(f"Model lokal dipilih: {preset} -> {MODEL_PRESETS[preset]}")
                self._update_model_source_ui()
            else:
                self.model_source_var.set("huggingface")
                self._update_model_source_ui()

    def sync_local_models_ui(self):
        """Sinkronkan ulang SEMUA folder model ke satu registry preset (tab Server)."""
        presets = sync_model_presets(log_fn=self.log)
        self.preset_combo["values"] = list(presets.keys())
        local_count = sum(1 for k in presets if k.startswith(LOCAL_MODEL_PREFIX))
        roots_text = "\n".join(
            f"  {'[OK]' if os.path.isdir(d) else '[--]'} {d}" for d in LOCAL_MODELS_DIRS)
        messagebox.showinfo(
            "Sync Model Lokal (Satu Server)",
            f"Sinkronisasi selesai.\n"
            f"Total preset tersedia: {len(presets)}\n"
            f"Model lokal ditemukan: {local_count}\n\n"
            f"Folder yang dipindai:\n{roots_text}"
        )

    def sync_codeassist_ui(self):
        """Sinkronkan modul NusaAi_codeassist ke server utama (tab Server).

        Menulis config bridge berisi endpoint server aktif (port & model yang
        sedang dikonfigurasi) sehingga ekstensi editor terhubung ke SATU server.
        """
        try:
            port = int(self.server_port.get())
        except ValueError:
            port = 8000
        model = (self.server_model_id.get().strip()
                 if self.model_source_var.get() == "huggingface"
                 else self.server_model_path.get().strip()) or None
        try:
            path = sync_codeassist_extension(port=port, model=model, log_fn=self.log)
        except FileNotFoundError as e:
            messagebox.showerror("Sync CodeAssist", str(e))
            return
        messagebox.showinfo(
            "Sync CodeAssist (Satu Server)",
            f"Modul nusa_ai Code Assist tersinkron ke server utama.\n\n"
            f"Server   : http://localhost:{port}\n"
            f"Model    : {model or '(mengikuti riwayat sinkronisasi)'}\n"
            f"Bridge   : {path}\n\n"
            f"Setting VS Code yang disarankan:\n"
            f"  nusa_aicodeassist.a2a.address = \"http://localhost:{port}\""
        )

    def browse_server_model(self):
        filetypes = [
            ("GGUF files", "*.gguf"),
            ("Safetensors", "*.safetensors"),
            ("PyTorch", "*.bin"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Pilih model lokal", filetypes=filetypes)
        if path:
            if path.endswith('.gguf'):
                self.server_model_path.set(path)
                # Jika file .gguf, otomatis sarankan runtime native (llama-server.exe)
                # atau fallback ke llama-cpp-python jika extensions/backends kosong
                self.runtime_backend_var.set(
                    "gguf-native" if self.ext_backends else "gguf-vulkan")
            else:
                folder = os.path.dirname(path)
                if os.path.exists(os.path.join(folder, "config.json")):
                    self.server_model_path.set(folder)
                    self.log(f"Folder model terdeteksi: {folder}")
                else:
                    messagebox.showerror("Error", "File model tidak didukung. Pilih folder yang berisi config.json atau file .gguf.")
        else:
            folder = filedialog.askdirectory(title="Pilih folder model (berisi config.json)")
            if folder:
                self.server_model_path.set(folder)

    def start_server(self):
        if self.server_running:
            return

        source = self.model_source_var.get()
        if source == "local":
            model_path = self.server_model_path.get().strip()
            if not model_path:
                messagebox.showerror("Error", "Pilih model lokal terlebih dahulu.")
                return
            model_id = model_path
        else:
            model_id = self.server_model_id.get().strip()
            if not model_id:
                messagebox.showerror("Error", "Masukkan model ID Hugging Face.")
                return

        # Tolak preset API yang tidak bisa dimuat di server lokal sedini mungkin.
        if model_id == "ANTHROPIC_API" or "ANTHROPIC_API" in MODEL_PRESETS.get(
                self.model_preset_var.get(), ""):
            messagebox.showerror(
                "Error",
                "Preset 'Claude (API - tidak lokal)' tidak dapat dimuat secara lokal. "
                "Gunakan model lain (ID HF, path folder Transformers, atau .gguf).")
            return

        try:
            port = int(self.server_port.get())
        except ValueError:
            messagebox.showerror("Error", "Port harus berupa angka.")
            return

        # DISABLE tombol Start secara instan untuk mencegah double-click
        self.start_btn.config(state=DISABLED)

        try:
            # SELALU buat instance ServerThread BARU setiap kali Start ditekan
            self.server_thread = ServerThread(
                model_source=source,
                model_path_or_id=model_id,
                port=port,
                device=self.device_var.get(),
                log_queue=self.log_queue,
                use_mcp=bool(self.use_mcp_var.get()),
                runtime_backend=self.runtime_backend_var.get(),
                server_framework=self.server_framework_var.get(),
                ext_backends=self.ext_backends
            )
            
            # Eksekusi thread baru
            self.server_thread.start()
            
            self.server_running = True
            self.stop_btn.config(state=NORMAL)
            self.log(f"Server dimulai dengan model: {model_id} pada port {port} (Framework: {self.server_framework_var.get()})")
            self.root.after(2000, self._poll_server_state)
            
        except Exception as e:
            self.log(f"Gagal memulai server: {e}")
            self.start_btn.config(state=NORMAL)

    def stop_server(self):
        if not self.server_running:
            return
            
        if self.server_thread:
            # Hentikan server API (Flask/FastAPI) dan proses native
            # llama-server.exe dari extensions/backends (jika ada)
            self.server_thread.stop()
                    
        self.server_running = False
        
        # HAPUS referensi thread lama agar memori bersih 
        self.server_thread = None 
        
        self.start_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)
        self.log("Server dihentikan.")

    def _poll_server_state(self):
        """Pantau kesehatan thread server; pulihkan tombol bila gagal dimuat
        atau thread berhenti mendadak (mis. exception di dalam worker)."""
        t = getattr(self, "server_thread", None)
        if t is not None and self.server_running:
            st = getattr(t, "state", "starting")
            if st == "load-failed":
                self.server_running = False
                self.server_thread = None
                self.start_btn.config(state=NORMAL)
                self.stop_btn.config(state=DISABLED)
                self.log("Server gagal dimuat; status dimulai kembali menjadi siap.")
                return
            if st == "running" and not t.is_alive():
                # FastAPI/Flask server (loop utama thread) berhenti dengan sendirinya.
                self.server_running = False
                self.server_thread = None
                self.start_btn.config(state=NORMAL)
                self.stop_btn.config(state=DISABLED)
                self.log("Server berhenti dengan sendirinya.")
                return
        
        if hasattr(self, "root") and getattr(self.root, "winfo_exists", lambda: False)():
            # Lanjutkan polling hanya selama jendela masih hidup.
            self.root.after(2000, self._poll_server_state)

    def _on_window_close(self):
        """Bersihkan proses server & llama-server.exe sebelum aplikasi ditutup."""
        try:
            if self.server_running and self.server_thread is not None:
                self.server_thread.stop()
                self.server_running = False
                self.server_thread = None
        except Exception as e:
            self.log(f"Peringatan saat menutup aplikasi: {e}")
        try:
            self.root.destroy()
        except Exception:
            pass

    # ------------------ CHAT TAB (AGENT CHAT KOMPREHENSIF) ------------------
    def _create_chat_tab(self):
        chat_tab = ttk.Frame(self.notebook)
        self.notebook.add(chat_tab, text="Chat")

        # ---------- TOOLBAR: MODE AGENT + SKILLS + TOOLS ----------
        toolbar = ttk.LabelFrame(chat_tab, text="Agent: Skills & Tools")
        toolbar.pack(fill=X, padx=5, pady=(5, 2))

        self.chat_agent_mode_var = IntVar(value=1)
        ttk.Checkbutton(toolbar, text="Mode Agent", variable=self.chat_agent_mode_var,
                        command=self._update_chat_tools_state).grid(row=0, column=0, padx=5, sticky=W)

        ttk.Label(toolbar, text="Skill Fokus:").grid(row=0, column=1, padx=(10, 0), sticky=W)
        self.chat_skill_var = StringVar(value="Semua Skill")
        self.chat_skill_combo = ttk.Combobox(toolbar, textvariable=self.chat_skill_var,
                                             width=17, state="readonly",
                                             values=["Semua Skill"] + CHAT_SKILLS)
        self.chat_skill_combo.grid(row=0, column=2, padx=5, sticky=W)

        tools_box = ttk.Frame(toolbar)
        tools_box.grid(row=0, column=3, padx=(15, 0), sticky=W)
        ttk.Label(tools_box, text="Tools:").pack(side=LEFT)
        self.chat_tool_search_var = IntVar(value=1)
        self.chat_tool_calc_var = IntVar(value=1)
        self.chat_tool_file_var = IntVar(value=1)
        self.chat_tool_sysinfo_var = IntVar(value=0)
        ttk.Checkbutton(tools_box, text="Web Search", variable=self.chat_tool_search_var).pack(side=LEFT, padx=3)
        ttk.Checkbutton(tools_box, text="Kalkulator", variable=self.chat_tool_calc_var).pack(side=LEFT, padx=3)
        ttk.Checkbutton(tools_box, text="File System", variable=self.chat_tool_file_var).pack(side=LEFT, padx=3)
        ttk.Checkbutton(tools_box, text="Info Sistem", variable=self.chat_tool_sysinfo_var).pack(side=LEFT, padx=3)

        plugin_names = ", ".join(f"{p['owner']}/{p['name']}"
                                 for p in getattr(self, "ext_plugins", [])) or "(tidak ada)"
        ttk.Label(toolbar, foreground="gray",
                  text=(f"Plugins extensions: {plugin_names}  |  Perintah: /help /calc "
                        f"/search /tree /sysinfo /files /skills /save /clear")).grid(
            row=1, column=0, columnspan=4, sticky=W, padx=5, pady=(2, 0))

        # ---------- LAMPIRAN: ADD FILES / ADD FOLDER PROYEK ----------
        attach_frame = ttk.LabelFrame(chat_tab, text="Konteks Terlampir (Add Files / Add Folder Proyek)")
        attach_frame.pack(fill=X, padx=5, pady=2)

        att_btns = ttk.Frame(attach_frame)
        att_btns.pack(fill=X, padx=5, pady=(4, 0))
        ttk.Button(att_btns, text="Add Files", command=self._attach_chat_files).pack(side=LEFT, padx=2)
        ttk.Button(att_btns, text="Add Folder Proyek", command=self._attach_chat_folder).pack(side=LEFT, padx=2)
        ttk.Button(att_btns, text="Hapus Terpilih", command=self._remove_selected_attachments).pack(side=LEFT, padx=2)
        ttk.Button(att_btns, text="Bersihkan", command=self._clear_chat_attachments).pack(side=LEFT, padx=2)

        self.chat_attach_list = Listbox(attach_frame, height=3)
        self.chat_attach_list.pack(fill=X, padx=5, pady=(2, 4))

        # ---------- RIWAYAT CHAT (BERWARNA PER PERAN) ----------
        self.chat_history = scrolledtext.ScrolledText(chat_tab, state='disabled', wrap=WORD)
        self.chat_history.pack(fill=BOTH, expand=True, padx=5, pady=2)
        for tag, color in (("user", "#0055cc"), ("model", "#007700"),
                           ("system", "#888888"), ("tool", "#7700aa"),
                           ("error", "#cc0000")):
            self.chat_history.tag_configure(tag, foreground=color)

        # ---------- STATUS BAR ----------
        self.chat_status_var = StringVar(
            value="Siap | Server: MATI | Lampiran: 0 | Riwayat: 0 pesan")
        ttk.Label(chat_tab, textvariable=self.chat_status_var,
                  foreground="gray").pack(fill=X, padx=6)

        # ---------- INPUT MULTI-BARIS ----------
        input_frame = ttk.Frame(chat_tab)
        input_frame.pack(fill=X, padx=5, pady=(2, 5))

        in_scroll = ttk.Frame(input_frame)
        in_scroll.pack(side=LEFT, fill=BOTH, expand=True)
        in_bar = ttk.Scrollbar(in_scroll)
        in_bar.pack(side=RIGHT, fill=Y)
        self.chat_input = Text(in_scroll, height=3, wrap=WORD, yscrollcommand=in_bar.set)
        self.chat_input.pack(side=LEFT, fill=BOTH, expand=True)
        in_bar.config(command=self.chat_input.yview)
        self.chat_input.bind("<Return>", self._on_chat_return)
        self.chat_input.bind("<KeyRelease>", lambda e: self._update_chat_status())

        btn_col = ttk.Frame(input_frame)
        btn_col.pack(side=LEFT, fill=Y, padx=(5, 0))
        self.chat_send_btn = ttk.Button(btn_col, text="Kirim", command=self._send_chat)
        self.chat_send_btn.pack(fill=X)
        self.chat_stop_btn = ttk.Button(btn_col, text="Stop", command=self._stop_chat,
                                        state=DISABLED)
        self.chat_stop_btn.pack(fill=X, pady=(4, 0))
        ttk.Button(btn_col, text="Export", command=self._export_chat).pack(fill=X, pady=(4, 0))
        ttk.Button(btn_col, text="Bersihkan", command=self._clear_chat).pack(fill=X, pady=(4, 0))

        self.chat_input.focus()
        self._update_chat_tools_state()
        self._append_chat(
            "System",
            "Nusa Ai Agent Chat siap.\n"
            "- Mode Agent menyematkan system prompt coding agent + skill fokus + tools.\n"
            "- Lampirkan konteks lewat 'Add Files' / 'Add Folder Proyek'.\n"
            "- Tools aktif dipakai otomatis saat relevan (web search, kalkulator,\n"
            "  file system, info sistem) dan hasilnya disematkan ke prompt.\n"
            "- Ketik /help untuk daftar perintah.",
            "system")

    def _on_chat_return(self, event):
        """Enter = kirim pesan, Ctrl+Enter = baris baru."""
        if int(event.state) & 0x0004:  # Ctrl ditekan -> biarkan newline
            return None
        self._send_chat()
        return "break"

    def _update_chat_tools_state(self):
        """Info saat Mode Agent diubah (tools & skill hanya berlaku di mode agent)."""
        if self.chat_agent_mode_var.get():
            self._append_chat("System",
                              "Mode Agent AKTIF: system prompt coding agent, skill fokus, "
                              "dan tools disematkan ke setiap permintaan.", "system")
        else:
            self._append_chat("System",
                              "Mode Agent NONAKTIF: chat biasa dengan riwayat singkat.", "system")

    def _chat_flags(self):
        """Snapshot status toggle chat (dibaca di main thread, dipakai di worker)."""
        return {
            "agent": bool(self.chat_agent_mode_var.get()),
            "search": bool(self.chat_tool_search_var.get()),
            "calc": bool(self.chat_tool_calc_var.get()),
            "file": bool(self.chat_tool_file_var.get()),
            "sysinfo": bool(self.chat_tool_sysinfo_var.get()),
            "skill": self.chat_skill_var.get(),
        }

    def _update_chat_status(self, busy=False):
        """Perbarui status bar chat."""
        server = "JALAN" if self.server_running else "MATI"
        mode = "Agent" if self.chat_agent_mode_var.get() else "Chat"
        if busy:
            self.chat_status_var.set(
                f"Memproses... | Mode: {mode} | Server: {server} | "
                f"Lampiran: {len(self.chat_attachments)}")
            return
        try:
            chars = len(self.chat_input.get("1.0", "end-1c"))
        except Exception:
            chars = 0
        self.chat_status_var.set(
            f"Mode: {mode} | Server: {server} | Lampiran: {len(self.chat_attachments)} | "
            f"Pesan: {len(self.chat_messages)} | ~{chars} karakter | "
            f"Enter=kirim, Ctrl+Enter=baris baru")

    def _stop_chat(self):
        """Batalkan generasi yang sedang berjalan (respons akan diabaikan)."""
        if self._chat_busy:
            self._chat_cancelled = True
            self._append_chat("System",
                              "Stop diminta. Respons yang datang akan diabaikan.", "system")

    def _send_chat(self):
        text = self.chat_input.get("1.0", "end-1c").strip()
        if not text:
            return
        self.chat_input.delete("1.0", "end")
        self._update_chat_status()

        # Perintah slash diproses lokal (tanpa memanggil model)
        if text.startswith("/"):
            self._handle_chat_command(text)
            return

        self._append_chat("Anda", text, "user")
        self.chat_messages.append({"role": "user", "content": text,
                                   "time": time.strftime("%H:%M:%S")})
        self._save_chat_session()

        if not self.server_running:
            self._append_chat("System",
                              "Server belum berjalan. Mulai server dulu di tab Server.", "system")
            return
        if self._chat_busy:
            self._append_chat("System",
                              "Generasi sebelumnya masih berjalan. Klik 'Stop' lalu coba lagi.", "system")
            return

        flags = self._chat_flags()
        # Snapshot parameter inferensi & port SEBELUM masuk worker thread.
        # Variabel Tk (StringVar/IntVar/DoubleVar) hanya boleh dibaca di main
        # thread; membaca dari worker menghasilkan error acak (TclError).
        gen_params = {
            "port": self.server_port.get().strip() or "8000",
            "max_tokens": self.max_tokens_var.get(),
            "temperature": self.temperature_var.get(),
            "repetition_penalty": self.repetition_penalty_var.get(),
            "top_p": self.top_p_var.get(),
            "timeout": self.request_timeout_var.get(),
        }
        self._chat_busy = True
        self._chat_cancelled = False
        self.chat_send_btn.config(state=DISABLED)
        self.chat_stop_btn.config(state=NORMAL)
        self._update_chat_status(busy=True)

        threading.Thread(target=self._chat_worker, args=(text, flags, gen_params),
                         daemon=True).start()

    def _chat_worker(self, user_text, flags, gen_params):
        """Worker chat: susun konteks (skills, tools, lampiran), jalankan tools,
        lalu panggil server inferensi — semuanya di luar main thread GUI.
        `gen_params` adalah snapshot parameter dari main thread (anti-race)."""
        params = gen_params or {
            "port": "8000", "max_tokens": 256, "temperature": 0.7,
            "repetition_penalty": 1.1, "top_p": 0.95, "timeout": 300,
        }
        try:
            prompt = self._build_chat_prompt(user_text, flags)

            # Jalankan tools lokal yang relevan; hasil disematkan ke prompt
            tools_out = self._run_chat_tools(user_text, flags)
            if tools_out:
                prompt += ("\n\n[HASIL TOOLS]\n" + tools_out +
                           "\n(Gunakan hasil tools di atas bila relevan.)")

            if self._chat_cancelled:
                return

            if requests is None:
                raise RuntimeError("Library 'requests' tidak tersedia.")

            response = requests.post(
                f"http://localhost:{params['port']}/generate",
                json={"prompt": prompt,
                      "max_tokens": params["max_tokens"],
                      "temperature": params["temperature"],
                      "repetition_penalty": params["repetition_penalty"],
                      "top_p": params["top_p"]},
                timeout=params["timeout"],
            ).json()
            answer = response.get("response", "Tidak ada respons")
            if self._chat_cancelled:
                self.root.after(0, lambda: self._append_chat(
                    "System", "Respons diabaikan karena tombol Stop ditekan.", "system"))
                return
            self.root.after(0, lambda: self._finish_chat_response(answer, prompt))
        except Exception as e:
            if not self._chat_cancelled:
                err = str(e)
                self.root.after(0, lambda: self._append_chat("Error", err, "error"))
        finally:
            self._chat_busy = False
            self.root.after(0, self._chat_set_idle)

    def _chat_set_idle(self):
        self.chat_send_btn.config(state=NORMAL)
        self.chat_stop_btn.config(state=DISABLED)
        self._update_chat_status()

    def _finish_chat_response(self, answer, prompt):
        self._append_chat("Model", answer, "model")
        self.chat_messages.append({"role": "assistant", "content": answer,
                                   "time": time.strftime("%H:%M:%S")})
        tokens = max(1, (len(prompt) + len(answer)) // 4)
        self._update_chat_status()
        self._append_chat("System", f"[~{tokens} token terpakai pada giliran ini]", "system")
        self._save_chat_session()

    def _append_chat(self, sender, message, tag="user"):
        """Tampilkan pesan di riwayat chat dengan warna per peran (thread-safe)."""
        def _do():
            self.chat_history.config(state='normal')
            stamp = time.strftime("%H:%M:%S")
            self.chat_history.insert(END, f"[{stamp}] {sender}:\n", tag)
            self.chat_history.insert(END, f"{message}\n\n", tag)
            self.chat_history.config(state='disabled')
            self.chat_history.see(END)
        try:
            self.root.after(0, _do)
        except RuntimeError:
            pass  # aplikasi sudah ditutup

    # ---------- CHAT: SKILLS, TOOLS & PENYUSUNAN KONTEKS ----------
    def _get_skill_value(self, name):
        """Ambil nilai skill dari tab Skills (fallback 0.5 jika belum ada)."""
        try:
            var = getattr(self, "skill_vars", {}).get(name)
            return float(var.get()) if var is not None else 0.5
        except Exception:
            return 0.5

    def _build_agent_skills_text(self, flags=None):
        """Susun teks persona skill agent (tersambung ke slider tab Skills)."""
        if flags is None:
            flags = self._chat_flags()
        focus = flags.get("skill", "Semua Skill")
        lines = []
        for skill in CHAT_SKILLS:
            if focus not in ("Semua Skill", skill):
                continue
            val = self._get_skill_value(skill)
            label = "tinggi" if val >= 0.67 else ("sedang" if val >= 0.34 else "rendah")
            lines.append(f"- {skill}: {val:.2f} ({label})")
        if not lines:
            return ""
        return ("[SKILL AGENT] (skala 0.0-1.0)\n" + "\n".join(lines)
                + f"\nSkill fokus: {focus}. Selaraskan gaya jawaban dengan skill di atas.")

    def _build_attachment_context(self, flags=None, max_total=30000):
        """Susun konteks dari file/folder yang dilampirkan di chat."""
        if flags is None:
            flags = self._chat_flags()
        chunks = []
        total = 0
        for att in self.chat_attachments:
            if att["type"] == "file":
                text = tool_read_file(att["path"], max_chars=CHAT_MAX_FILE_CHARS)
                if text:
                    chunks.append(f"=== FILE: {att['path']} ===\n{text}")
            else:  # folder proyek
                tree = tool_folder_tree(att["path"], max_files=CHAT_MAX_TREE_FILES)
                chunks.append(f"=== STRUKTUR FOLDER PROYEK: {att['path']} ===\n{tree}")
                if flags["file"]:
                    preview = tool_folder_files_preview(
                        att["path"], max_files=CHAT_MAX_FOLDER_FILES)
                    if preview:
                        chunks.append(preview)
            total = sum(len(c) for c in chunks)
            if total >= max_total:
                break
        if not chunks:
            return ""
        out = "\n\n".join(chunks)
        if len(out) > max_total:
            out = out[:max_total] + "\n... (konteks dipotong karena batas ukuran)"
        return out

    def _build_chat_prompt(self, user_text, flags=None):
        """Bangun prompt akhir: system prompt agent + skills + konteks + riwayat."""
        if flags is None:
            flags = self._chat_flags()
        parts = []
        if flags["agent"]:
            parts.append("[SYSTEM]\n" + CODING_AGENT_SYSTEM_PROMPT)
            tools_on = [name for name, key in (
                ("web-search", "search"), ("kalkulator", "calc"),
                ("file-system", "file"), ("info-sistem", "sysinfo")) if flags[key]]
            parts.append("[TOOLS AKTIF] " + (", ".join(tools_on) or "(tidak ada)")
                         + ". Jalankan tools bila relevan dengan permintaan pengguna.")
            skills_text = self._build_agent_skills_text(flags)
            if skills_text:
                parts.append(skills_text)
            parts.append("[KONTEKS TERLAMPIR]\n"
                         + (self._build_attachment_context(flags)
                            or "(belum ada file/folder terlampir)"))

        # Riwayat percakapan (maks 6 pesan terakhir sebelum pesan ini)
        history = self.chat_messages[:-1][-6:]
        if history:
            parts.append("[RIWAYAT PERCAKAPAN]")
            for msg in history:
                who = "Pengguna" if msg["role"] == "user" else "Asisten"
                content = msg["content"]
                if len(content) > 800:
                    content = content[:800] + " ..."
                parts.append(f"{who}: {content}")

        parts.append(f"Pengguna: {user_text}")
        parts.append("Asisten:")
        return "\n\n".join(parts)

    def _run_chat_tools(self, user_text, flags=None):
        """Jalankan tools lokal yang relevan; kembalikan teks hasil untuk prompt.
        Dipanggil dari worker thread; hasil tampil di chat dengan tag 'tool'."""
        if flags is None:
            flags = self._chat_flags()
        extra = []

        # Info sistem: disematkan ke konteks bila tool diaktifkan
        if flags["agent"] and flags["sysinfo"]:
            info = tool_system_info()
            self._append_chat("Tool [Info Sistem]", info, "tool")
            extra.append("[INFO SISTEM]\n" + info)

        # Kalkulator otomatis: pesan berupa ekspresi matematika murni
        if flags["agent"] and flags["calc"]:
            expr = user_text.strip().rstrip("?=").strip()
            if (re.fullmatch(r"[\d\s\.\+\-\*/\(\)%\^]+", expr)
                    and re.search(r"\d", expr)
                    and re.search(r"[\+\-\*/%\^]", expr)):
                try:
                    result = tool_calculator(expr.replace("^", "**"))
                    self._append_chat(f"Tool [Kalkulator] {expr}", result, "tool")
                    extra.append(f"[HASIL KALKULATOR] {result}")
                except Exception as e:
                    self._append_chat("Tool [Kalkulator]",
                                      f"Gagal menghitung '{expr}': {e}", "error")

        # Web search otomatis: pesan diawali "cari .../search ..."
        if flags["agent"] and flags["search"]:
            m = re.match(r"^(?:cari|search|googling)\s+(?:tentang\s+|informasi\s+)?(.+)",
                         user_text.strip(), re.IGNORECASE)
            if m:
                query = m.group(1)
                self._append_chat("Tool [Web Search]", f"Mencari: {query} ...", "system")
                try:
                    result = tool_web_search(query)
                    self._append_chat(f"Tool [Web Search] '{query}'", result, "tool")
                    extra.append(f"[HASIL WEB SEARCH '{query}']\n{result}")
                except Exception as e:
                    self._append_chat("Tool [Web Search]", f"Gagal: {e}", "error")

        return "\n\n".join(extra)

    # ---------- CHAT: PERINTAH SLASH ----------
    def _handle_chat_command(self, text):
        """Proses perintah slash lokal (tanpa memanggil model)."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/help", "/?"):
            self._append_chat("System", CHAT_HELP_TEXT, "system")
        elif cmd == "/clear":
            self._clear_chat()
        elif cmd == "/save":
            self._export_chat()
        elif cmd == "/calc":
            if not arg:
                self._append_chat("System", "Pemakaian: /calc 2*(3+4)^2", "system")
            else:
                try:
                    self._append_chat("Tool [Kalkulator]",
                                      tool_calculator(arg.replace("^", "**")), "tool")
                except Exception as e:
                    self._append_chat("Error", f"Kalkulator: {e}", "error")
        elif cmd == "/search":
            if not arg:
                self._append_chat("System", "Pemakaian: /search <query>", "system")
            else:
                self._append_chat("Tool [Web Search]", f"Mencari: {arg} ...", "system")

                def _search_worker(q):
                    try:
                        self._append_chat(f"Tool [Web Search] '{q}'",
                                          tool_web_search(q), "tool")
                    except Exception as e:
                        self._append_chat("Error", f"Web search gagal: {e}", "error")

                threading.Thread(target=_search_worker, args=(arg,), daemon=True).start()
        elif cmd == "/tree":
            folders = [a for a in self.chat_attachments if a["type"] == "folder"]
            if not folders:
                self._append_chat("System",
                                  "Tidak ada folder proyek terlampir. "
                                  "Gunakan tombol 'Add Folder Proyek'.", "system")
            else:
                for f in folders:
                    self._append_chat(f"Tool [Tree] {f['path']}",
                                      tool_folder_tree(f["path"],
                                                       max_files=CHAT_MAX_TREE_FILES),
                                      "tool")
        elif cmd == "/sysinfo":
            self._append_chat("Tool [Info Sistem]", tool_system_info(), "tool")
        elif cmd == "/files":
            if not self.chat_attachments:
                self._append_chat("System", "Belum ada file/folder terlampir.", "system")
            else:
                listing = "\n".join(f"[{a['type']}] {a['path']}"
                                    for a in self.chat_attachments)
                self._append_chat("System", f"Konteks terlampir:\n{listing}", "system")
        elif cmd == "/skills":
            self._append_chat("System",
                              self._build_agent_skills_text() or "(tidak ada skill)",
                              "system")
        else:
            self._append_chat("System",
                              f"Perintah tidak dikenal: {cmd}. Ketik /help untuk bantuan.",
                              "system")

    def _attach_chat_files(self):
        """Tambah satu/beberapa file ke konteks chat."""
        paths = filedialog.askopenfilenames(title="Tambah file ke konteks chat")
        added = 0
        for p in paths:
            if p and p not in [a["path"] for a in self.chat_attachments]:
                self.chat_attachments.append({"type": "file", "path": p})
                added += 1
        if added:
            self._refresh_attachment_list()
            self._append_chat("System",
                              f"{added} file ditambahkan ke konteks chat. "
                              "Isi file akan disematkan otomatis ke prompt agent.", "system")

    def _attach_chat_folder(self):
        """Tambah folder proyek ke konteks chat (struktur + cuplikan file penting)."""
        folder = filedialog.askdirectory(title="Tambah folder proyek ke konteks chat")
        if folder:
            if folder not in [a["path"] for a in self.chat_attachments]:
                self.chat_attachments.append({"type": "folder", "path": folder})
                self._refresh_attachment_list()
                self._append_chat(
                    "System",
                    f"Folder proyek ditambahkan: {folder}\n"
                    "Struktur folder disematkan; cuplikan file kode/dok penting "
                    "ikut saat Mode Agent + File System aktif.", "system")

    def _remove_selected_attachments(self):
        for idx in sorted(self.chat_attach_list.curselection(), reverse=True):
            del self.chat_attachments[idx]
        self._refresh_attachment_list()

    def _clear_chat_attachments(self):
        self.chat_attachments.clear()
        self._refresh_attachment_list()
        self._append_chat("System", "Semua lampiran konteks dibersihkan.", "system")

    def _refresh_attachment_list(self):
        self.chat_attach_list.delete(0, END)
        for a in self.chat_attachments:
            icon = "[DIR]" if a["type"] == "folder" else "[FILE]"
            self.chat_attach_list.insert(END, f"{icon} {a['path']}")
        self._update_chat_status()

    def _clear_chat(self):
        """Bersihkan riwayat chat (layar + memori)."""
        self.chat_messages.clear()
        self.chat_history.config(state='normal')
        self.chat_history.delete(1.0, END)
        self.chat_history.config(state='disabled')
        self._save_chat_session()
        self._append_chat("System", "Riwayat chat dibersihkan.", "system")
        self._update_chat_status()

    def _export_chat(self):
        """Ekspor percakapan ke file Markdown."""
        if not self.chat_messages:
            messagebox.showinfo("Info", "Belum ada percakapan untuk diekspor.")
            return
        path = filedialog.asksaveasfilename(
            title="Ekspor percakapan", defaultextension=".md",
            initialfile=f"chat_{time.strftime('%Y%m%d_%H%M%S')}.md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Nusa Ai LLM Studio - Riwayat Chat\n\n")
                f.write(f"Diekspor: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for msg in self.chat_messages:
                    who = "## Anda" if msg["role"] == "user" else "## Model"
                    f.write(f"{who} ({msg.get('time', '')}):\n\n{msg['content']}\n\n---\n\n")
            self.log(f"Chat diekspor ke: {path}")
            self._append_chat("System", f"Chat diekspor ke: {path}", "system")
        except Exception as e:
            self._append_chat("Error", f"Gagal mengekspor chat: {e}", "error")

    def _save_chat_session(self):
        """Simpan sesi chat terakhir ke config/chat_history.json (auto-save)."""
        try:
            cfg_dir = os.path.join(STUDIO_DIR, "config")
            os.makedirs(cfg_dir, exist_ok=True)
            with open(os.path.join(cfg_dir, "chat_history.json"), "w",
                      encoding="utf-8") as f:
                json.dump(self.chat_messages[-100:], f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # ------------------ TRAINING TAB ------------------
    def _create_training_tab(self):
        train_tab = ttk.Frame(self.notebook)
        self.notebook.add(train_tab, text="Training")

        # ---------- PANEL CODING AGENT ----------
        coder_frame = ttk.LabelFrame(train_tab, text="Coding Agent Trainer (Model Spesifik Koding)")
        coder_frame.pack(fill=X, padx=10, pady=10)

        ttk.Label(coder_frame, text="Base Model Coding:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        self.coder_preset_var = StringVar(value=list(CODING_AGENT_PRESETS.keys())[0])
        coder_combo = ttk.Combobox(coder_frame, textvariable=self.coder_preset_var,
                                   values=list(CODING_AGENT_PRESETS.keys()), width=45, state="readonly")
        coder_combo.grid(row=0, column=1, padx=5, pady=5, sticky=W)
        ttk.Button(coder_frame, text="Gunakan Preset Ini",
                   command=self.use_coding_preset).grid(row=0, column=2, padx=5)

        ttk.Button(coder_frame, text="1) Bangun Dataset dari Folder data/",
                   command=self.build_coding_dataset).grid(row=1, column=0, padx=5, pady=5, sticky=W)
        ttk.Button(coder_frame, text="2) Mulai Training Coding Agent",
                   command=self.start_coding_agent_training).grid(row=1, column=1, padx=5, pady=5, sticky=W)

        self.coding_agent_info_var = StringVar(
            value=(f"Dataset dibangun otomatis dari semua file di '{DEFAULT_DATA_DIR}' "
                   "(md/json/jsonl/source code) menjadi coding_agent_dataset.jsonl, "
                   "lalu model coding spesifik dilatih dan hasilnya tersinkron ke models/."))
        ttk.Label(coder_frame, textvariable=self.coding_agent_info_var,
                  foreground="blue", wraplength=780, justify=LEFT).grid(
            row=2, column=0, columnspan=3, padx=5, pady=5, sticky=W)

        # ---------- FINE-TUNING UMUM ----------
        frame = ttk.LabelFrame(train_tab, text="Fine-tuning Model")
        frame.pack(fill=X, padx=10, pady=10)

        ttk.Label(frame, text="Base Model (HF ID / Path):").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        self.train_base_model = StringVar(value="Qwen/Qwen2.5-Coder-0.5B-Instruct")
        ttk.Entry(frame, textvariable=self.train_base_model, width=50).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Dataset File (.txt/.jsonl/.json):").grid(row=1, column=0, padx=5, pady=5, sticky=W)
        self.train_dataset_path = StringVar(value="")
        ttk.Entry(frame, textvariable=self.train_dataset_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_dataset_file).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(frame, text="Output Directory:").grid(row=2, column=0, padx=5, pady=5, sticky=W)
        default_output = os.path.join(LOCAL_MODELS_DIR, "nusaai-coding-agent")
        self.train_output_dir = StringVar(value=default_output)
        ttk.Entry(frame, textvariable=self.train_output_dir, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_output_dir).grid(row=2, column=2, padx=5, pady=5)

        ttk.Label(frame, text="Epochs:").grid(row=3, column=0, padx=5, pady=5, sticky=W)
        self.train_epochs = IntVar(value=3)
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.train_epochs, width=10).grid(row=3, column=1, padx=5, pady=5, sticky=W)

        ttk.Label(frame, text="Batch Size:").grid(row=4, column=0, padx=5, pady=5, sticky=W)
        self.train_batch_size = IntVar(value=2)
        ttk.Spinbox(frame, from_=1, to=64, textvariable=self.train_batch_size, width=10).grid(row=4, column=1, padx=5, pady=5, sticky=W)

        ttk.Label(frame, text="Learning Rate:").grid(row=5, column=0, padx=5, pady=5, sticky=W)
        self.train_lr = DoubleVar(value=5e-5)
        ttk.Entry(frame, textvariable=self.train_lr, width=10).grid(row=5, column=1, padx=5, pady=5, sticky=W)

        self.use_lora_var = IntVar(value=1)
        ttk.Checkbutton(frame, text="Gunakan LoRA (jika tersedia)", variable=self.use_lora_var).grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky=W)

        ttk.Button(frame, text="Mulai Training", command=self.start_training).grid(row=7, column=0, columnspan=2, pady=10)

        info_text = """
        Dataset format yang didukung:
        - .jsonl : Alpaca {"instruction","input","output"} / chat {"messages":[...]} / {"text":...}
        - .json  : array Alpaca atau pesan chat
        - .txt   : satu contoh per baris
        Hasil training otomatis tersinkron ke folder models/ agar bisa langsung
        dipilih lewat tombol "Sync Model Lokal" di tab Server.
        """
        ttk.Label(train_tab, text=info_text, justify=LEFT).pack(padx=10, pady=5)

    # ------------------ CODING AGENT HELPERS ------------------
    def use_coding_preset(self):
        preset = self.coder_preset_var.get()
        model_id = CODING_AGENT_PRESETS.get(preset)
        if not model_id:
            messagebox.showerror("Error", "Preset coding tidak dikenal.")
            return
        if model_id.lower().endswith(".gguf"):
            messagebox.showwarning(
                "Info",
                f"'{preset}' adalah model GGUF untuk inferensi agent (bukan base "
                "model fine-tuning). Training memakai preset Qwen2.5-Coder.")
            model_id = CODING_AGENT_PRESETS["Coder-Qwen2.5-Coder-0.5B-Instruct (ringan)"]
        self.train_base_model.set(model_id)
        self.train_output_dir.set(os.path.join(LOCAL_MODELS_DIR, "nusaai-coding-agent"))
        self.log(f"Preset Coding Agent aktif: {model_id}")

    def build_coding_dataset(self):
        if getattr(self, "_dataset_builder_thread", None) and self._dataset_builder_thread.is_alive():
            messagebox.showwarning("Info", "Pembuatan dataset sedang berjalan.")
            return

        data_dir = DEFAULT_DATA_DIR
        output_path = os.path.join(data_dir, "coding_agent_dataset.jsonl")

        def worker():
            try:
                self.log("[Coding Agent] Membangun dataset dari folder data/ ...")
                build_coding_agent_dataset(
                    data_dir=data_dir,
                    output_path=output_path,
                    log_fn=self.log,
                )
                self.root.after(0, lambda: self.train_dataset_path.set(output_path))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Sukses",
                    f"Dataset Coding Agent selesai dibuat:\n{output_path}\n\n"
                    "Path sudah terisi di kolom Dataset. Lanjutkan ke langkah 2."
                ))
            except Exception as e:
                self.log(f"[Coding Agent] ERROR membangun dataset: {e}")

        self._dataset_builder_thread = threading.Thread(target=worker, daemon=True)
        self._dataset_builder_thread.start()

    def start_coding_agent_training(self):
        """Training model spesifik coding agent (preset + dataset + system prompt)."""
        preset = self.coder_preset_var.get()
        model_id = CODING_AGENT_PRESETS.get(preset, "Qwen/Qwen2.5-Coder-0.5B-Instruct")

        dataset_path = os.path.join(DEFAULT_DATA_DIR, "coding_agent_dataset.jsonl")

        if not os.path.exists(dataset_path):
            if messagebox.askyesno(
                    "Dataset Belum Ada",
                    "File coding_agent_dataset.jsonl belum ada.\n"
                    "Bangun sekarang dari folder data/?"):
                self.build_coding_dataset()
            return

        self.use_coding_preset()
        self.train_base_model.set(model_id)
        self.train_dataset_path.set(dataset_path)
        self.train_output_dir.set(os.path.join(LOCAL_MODELS_DIR, "nusaai-coding-agent"))
        self._launch_training(system_prompt=CODING_AGENT_SYSTEM_PROMPT)

    def browse_dataset_file(self):
        path = filedialog.askopenfilename(title="Pilih dataset", filetypes=[("Text/JSONL", "*.txt *.jsonl"), ("All files", "*.*")])
        if path:
            self.train_dataset_path.set(path)

    def browse_output_dir(self):
        path = filedialog.askdirectory(title="Pilih folder output")
        if path:
            self.train_output_dir.set(path)

    def start_training(self):
        self._launch_training()

    def _launch_training(self, system_prompt=None):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Info", "Training sudah berjalan.")
            return

        base_model = self.train_base_model.get().strip()
        dataset_path = self.train_dataset_path.get().strip()
        output_dir = self.train_output_dir.get().strip()

        if not base_model or not dataset_path or not output_dir:
            messagebox.showerror("Error", "Base model, dataset, dan output directory wajib diisi.")
            return

        if not os.path.exists(dataset_path):
            messagebox.showerror("Error", "File dataset tidak ditemukan.")
            return

        # Pastikan model hasil training berakhir di folder models/ agar tersinkron
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(STUDIO_DIR, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.training_thread = TrainingThread(
            base_model=base_model,
            dataset_path=dataset_path,
            output_dir=output_dir,
            epochs=self.train_epochs.get(),
            batch_size=self.train_batch_size.get(),
            lr=self.train_lr.get(),
            use_lora=bool(self.use_lora_var.get()),
            log_queue=self.log_queue,
            system_prompt=system_prompt,
        )
        self.training_thread.start()
        self.log("Training dimulai. Lihat progress di tab Log.")

    # ------------------ DATASET TAB ------------------
    def _create_dataset_tab(self):
        dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(dataset_tab, text="Dataset")

        frame = ttk.LabelFrame(dataset_tab, text="Hugging Face Dataset Downloader")
        frame.pack(fill=X, padx=10, pady=10)

        ttk.Label(frame, text="Dataset ID:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        self.dataset_entry = ttk.Entry(frame, width=50)
        self.dataset_entry.grid(row=0, column=1, padx=5, pady=5)
        self.dataset_entry.insert(0, "wikitext")

        ttk.Label(frame, text="Split (opsional):").grid(row=1, column=0, padx=5, pady=5, sticky=W)
        self.split_entry = ttk.Entry(frame, width=30)
        self.split_entry.grid(row=1, column=1, padx=5, pady=5)
        self.split_entry.insert(0, "train")

        download_btn = ttk.Button(frame, text="Download", command=self.download_dataset)
        download_btn.grid(row=2, column=0, columnspan=2, pady=10)

        self.dataset_log = scrolledtext.ScrolledText(dataset_tab, state='disabled', height=10)
        self.dataset_log.pack(fill=BOTH, expand=True, padx=5, pady=5)

    def download_dataset(self):
        dataset_id = self.dataset_entry.get().strip()
        split = self.split_entry.get().strip()
        if not dataset_id:
            messagebox.showerror("Error", "Dataset ID tidak boleh kosong")
            return

        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
            self._log_dataset(f"Mengunduh dataset {dataset_id}...")
            dataset = load_dataset(dataset_id, split=split if split else None)
            self._log_dataset(f"Dataset berhasil diunduh! Jumlah baris: {len(dataset)}")
            save_dir = os.path.join(STUDIO_DIR, "datasets", dataset_id.replace("/", "_"))
            os.makedirs(save_dir, exist_ok=True)
            dataset.save_to_disk(save_dir)
            self._log_dataset(f"Disimpan di: {save_dir}")
        except ImportError:
            self._log_dataset("ERROR: Library 'datasets' belum diinstal. Jalankan: pip install datasets")
        except Exception as e:
            self._log_dataset(f"ERROR: {str(e)}")

    def _log_dataset(self, message):
        self.dataset_log.config(state='normal')
        self.dataset_log.insert(END, message + "\n")
        self.dataset_log.config(state='disabled')
        self.dataset_log.see(END)

    # ------------------ SETTINGS TAB ------------------
    def _create_settings_tab(self):
        settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(settings_tab, text="Runtime Settings")

        frame = ttk.LabelFrame(settings_tab, text="Parameter Inferensi")
        frame.pack(fill=X, padx=10, pady=10)

        ttk.Label(frame, text="Max Tokens:").grid(row=0, column=0, sticky=W, padx=5, pady=5)
        ttk.Spinbox(frame, from_=10, to=2048, textvariable=self.max_tokens_var, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(frame, text="Temperature:").grid(row=1, column=0, sticky=W, padx=5, pady=5)
        ttk.Scale(frame, from_=0.0, to=2.0, variable=self.temperature_var, orient=HORIZONTAL, length=200).grid(row=1, column=1, padx=5, sticky=W)

        ttk.Label(frame, text="Repetition Penalty:").grid(row=2, column=0, sticky=W, padx=5, pady=5)
        ttk.Scale(frame, from_=1.0, to=2.0, variable=self.repetition_penalty_var, orient=HORIZONTAL, length=200).grid(row=2, column=1, padx=5, sticky=W)

        ttk.Label(frame, text="Top P:").grid(row=3, column=0, sticky=W, padx=5, pady=5)
        ttk.Scale(frame, from_=0.1, to=1.0, variable=self.top_p_var, orient=HORIZONTAL, length=200).grid(row=3, column=1, padx=5, sticky=W)

        ttk.Label(frame, text="Request Timeout (detik):").grid(row=4, column=0, sticky=W, padx=5, pady=5)
        ttk.Spinbox(frame, from_=30, to=3600, textvariable=self.request_timeout_var, width=10).grid(row=4, column=1, padx=5, sticky=W)

        self.settings_label = ttk.Label(settings_tab, text="", foreground="blue")
        self.settings_label.pack(pady=5)
        self._update_settings_label()

        self.temperature_var.trace_add("write", lambda *args: self._update_settings_label())
        self.repetition_penalty_var.trace_add("write", lambda *args: self._update_settings_label())
        self.top_p_var.trace_add("write", lambda *args: self._update_settings_label())
        self.max_tokens_var.trace_add("write", lambda *args: self._update_settings_label())
        self.request_timeout_var.trace_add("write", lambda *args: self._update_settings_label())

    def _update_settings_label(self):
        try:
            # Safely attempt to get the values
            max_tokens = self.max_tokens_var.get()
            temp = self.temperature_var.get()
            rep_pen = self.repetition_penalty_var.get()
            top_p = self.top_p_var.get()
            timeout = self.request_timeout_var.get()

            # Update the label if successful
            self.settings_label.config(
                text=f"Max Tokens: {max_tokens} | Temperature: {temp:.2f} | "
                     f"Repetition Penalty: {rep_pen:.2f} | Top P: {top_p:.2f} | "
                     f"Timeout: {timeout}s"
            )
        except Exception:
            # If a field is empty (raises TclError), do nothing and wait for valid input
            pass

    # ------------------ LOG TAB ------------------
    def _create_log_tab(self):
        log_tab = ttk.Frame(self.notebook)
        self.notebook.add(log_tab, text="Log")

        self.log_text = scrolledtext.ScrolledText(log_tab, state='disabled', height=20)
        self.log_text.pack(fill=BOTH, expand=True, padx=5, pady=5)

        clear_btn = ttk.Button(log_tab, text="Bersihkan Log", command=self.clear_log)
        clear_btn.pack(pady=5)

    def log(self, message):
        self.log_queue.put(message)

    def clear_log(self):
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, END)
        self.log_text.config(state='disabled')

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.config(state='normal')
                self.log_text.insert(END, msg + "\n")
                self.log_text.config(state='disabled')
                self.log_text.see(END)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

# ================== MAIN ==================
def _set_window_icon(root):
    """Pasang logo resmi sebagai ikon jendela/taskbar (fallback ke PNG/GIF)."""
    try:
        ico = LOGO_ICO
        if os.path.isfile(ico):
            try:
                root.iconbitmap(default=ico)
                root.iconbitmap(ico)
                return
            except Exception:
                pass
        # Fallback: PhotoImage (GIF didukung tanpa PIL; PNG butuh PIL/ImageTk)
        for cand, is_png in ((LOGO_GIF, False), (LOGO_PNG, True)):
            if not os.path.isfile(cand):
                continue
            try:
                if is_png:
                    try:
                        from PIL import Image, ImageTk
                        img = ImageTk.PhotoImage(Image.open(cand))
                    except Exception:
                        continue
                else:
                    img = PhotoImage(file=cand)
                root.iconphoto(True, img)
                root._app_icon_img = img  # pertahankan referensi (anti GC)
                return
            except Exception:
                continue
    except Exception:
        pass


def main():
    # Modus CLI terintegrasi: python NusaAi_LLM_Studio.py --cli ...
    # Ini hanya jalan pintas; program CLI utama adalah Nusa_Ai_cli.py.
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        cli_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "Nusa_Ai_cli.py")
        if os.path.isfile(cli_path):
            code_path = os.path.abspath(cli_path)
            sys.argv = [code_path] + sys.argv[2:]
            with open(code_path, "r", encoding="utf-8") as f:
                exec(compile(f.read(), code_path, "exec"), {"__name__": "__main__",
                                                            "__file__": code_path})
            return 0
        print("Nusa_Ai_cli.py tidak ditemukan di samping NusaAi_LLM_Studio.py.")
        return 1

    if not _import_gui():
        print("ERROR: Tkinter tidak tersedia di Python ini.")
        print("  - Jalankan GUI:      python NusaAi_LLM_Studio.py")
        print("  - Jalankan server/chat via CLI: python Nusa_Ai_cli.py serve ...")
        print("  - Install Tk di Linux, mis. : sudo apt install python3-tk")
        return 1

    root = Tk()
    root.title(APP_NAME)
    _set_window_icon(root)
    app = TrainingApp(root)
    root.protocol("WM_DELETE_WINDOW", app._on_window_close)  # type: ignore[reportPrivateUsage]
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
