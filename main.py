"""
WAVE — API grading foto sampah (FastAPI).

Jalankan lokal:
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoint:
    GET  /         -> info service
    GET  /health   -> status model (apakah file .pth termuat)
    POST /predict  -> upload foto (multipart field "file"), kembali JSON:
        {
          "jenis": "cardboard",
          "jenis_confidence": 0.83,
          "jenis_top": {"cardboard": 0.83, "glass": 0.10, ...},
          "kondisi": "utuh",
          "kondisi_confidence": 0.38,
          "grade": "A",
          "review_flag": true,
          "catatan": "Kepercayaan kondisi di bawah 60% — perlu verifikasi operator."
        }

Pipeline sama dengan demo Gradio: foto -> klasifikasi jenis (ResNet50,
6 kelas) -> klasifikasi kondisi (ResNet50 fine-tune, 3 kelas) -> grade
A/B/C.

Catatan loader (penting, dari pengalaman di notebook WAVE):
  - best_grade_model.pth BUKAN state dict polos, melainkan dict pembungkus
    (model_state_dict + optimizer + metadata), dan key-nya ber-prefix
    "network." karena model jenis dibungkus class dengan self.network.
  - Head classifier di checkpoint hasil generate ulang tersimpan sebagai
    Sequential (Dropout + Linear) sehingga key-nya fc.1.weight/fc.1.bias,
    sementara app memakai Linear polos (fc.weight/fc.bias).
  Loader di bawah meng-unwrap pembungkus, men-strip prefix "network.",
  dan memetakan fc.1.* -> fc.* supaya bobot head terlatih ikut termuat
  (bukan bobot acak). Jumlah missing/unexpected di-print ke log saat start.

Aturan grading:
  - utuh -> A, rusak_ringan -> B, rusak_berat -> C
  - jenis "trash" selalu grade C (aturan pipeline WAVE), tanpa dinilai kondisi
  - confidence < 60% -> review_flag true (perlu verifikasi operator)
"""

import io
import os

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------- config
TYPE_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
COND_CLASSES = ["utuh", "rusak_ringan", "rusak_berat"]
GRADE_MAP = {"utuh": "A", "rusak_ringan": "B", "rusak_berat": "C"}
REVIEW_THRESHOLD = 0.6

TYPE_CKPT = os.environ.get("WAVE_TYPE_CKPT", "best_grade_model.pth")
COND_CKPT = os.environ.get("WAVE_COND_CKPT", "best_kondisi_model.pth")

DEVICE = torch.device("cpu")

# Preprocessing standar ImageNet (samakan dengan transform training).
PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


# ---------------------------------------------------------------- loader
def load_state_dict(path):
    """Muat checkpoint apa pun formatnya: polos, pembungkus, atau head Sequential."""
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    cleaned = {}
    for key, value in sd.items():
        if key.startswith("network."):
            key = key[len("network."):]
        cleaned[key] = value
    # Head Sequential (Dropout + Linear) menyimpan classifier di fc.1.*,
    # sementara model app memakai Linear polos (fc.*). Petakan fc.1.* -> fc.*
    # supaya bobot head terlatih termuat; jangan timpa fc.* yang sudah ada.
    remapped = {}
    for key, value in cleaned.items():
        new_key = key
        if key.startswith("fc.") and len(key) > 3 and key[3].isdigit():
            candidate = "fc." + key.split(".", 2)[2]
            if candidate not in cleaned:
                new_key = candidate
        remapped[new_key] = value
    return remapped


def build_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(path, num_classes):
    """Muat model kalau file-nya ada; kalau tidak ada, kembalikan None."""
    if not os.path.exists(path):
        print(f"Checkpoint belum ada, dilewati: {path}")
        return None
    model = build_model(num_classes)
    state = load_state_dict(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"{path}: missing={missing} unexpected={unexpected}")
    model.to(DEVICE).eval()
    return model


# Kedua model opsional saat start; /predict memberi pesan jelas kalau belum ada.
TYPE_MODEL = load_model(TYPE_CKPT, len(TYPE_CLASSES))
COND_MODEL = load_model(COND_CKPT, len(COND_CLASSES))


# ---------------------------------------------------------------- app
app = FastAPI(title="WAVE — Grading Foto Sampah API")

# Dibuka untuk semua origin supaya PWA operator WAVE bisa memanggil langsung.
# Untuk produksi, ganti "*" dengan domain PWA yang sebenarnya.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "WAVE grading API",
        "endpoints": ["/health", "/predict"],
        "models": {
            "type_loaded": TYPE_MODEL is not None,
            "cond_loaded": COND_MODEL is not None,
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "type_model_loaded": TYPE_MODEL is not None,
        "cond_model_loaded": COND_MODEL is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if TYPE_MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model jenis belum tersedia — upload best_grade_model.pth dulu.",
        )
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")

    with torch.no_grad():
        tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
        type_probs = torch.softmax(TYPE_MODEL(tensor), dim=1)[0]
        type_idx = int(type_probs.argmax())
        jenis = TYPE_CLASSES[type_idx]
        type_conf = float(type_probs[type_idx])
        jenis_top = {
            label: round(float(p), 4) for label, p in zip(TYPE_CLASSES, type_probs)
        }

    notes = []
    if type_conf < REVIEW_THRESHOLD:
        notes.append("Kepercayaan jenis di bawah 60% — perlu verifikasi operator.")

    if jenis == "trash":
        # Aturan pipeline WAVE: sampah campuran hampir selalu grade C.
        return {
            "jenis": jenis,
            "jenis_confidence": round(type_conf, 4),
            "jenis_top": jenis_top,
            "kondisi": None,
            "kondisi_confidence": None,
            "grade": "C",
            "review_flag": type_conf < REVIEW_THRESHOLD,
            "catatan": "Jenis trash selalu grade C sesuai aturan pipeline WAVE.",
        }

    if COND_MODEL is None:
        return {
            "jenis": jenis,
            "jenis_confidence": round(type_conf, 4),
            "jenis_top": jenis_top,
            "kondisi": None,
            "kondisi_confidence": None,
            "grade": None,
            "review_flag": type_conf < REVIEW_THRESHOLD,
            "catatan": (
                "Model kondisi belum tersedia — baru jenis yang dinilai. "
                "Upload best_kondisi_model.pth untuk mengaktifkan grading."
            ),
        }

    with torch.no_grad():
        cond_probs = torch.softmax(COND_MODEL(tensor), dim=1)[0]
        cond_idx = int(cond_probs.argmax())
        kondisi = COND_CLASSES[cond_idx]
        cond_conf = float(cond_probs[cond_idx])
    if cond_conf < REVIEW_THRESHOLD:
        notes.append("Kepercayaan kondisi di bawah 60% — perlu verifikasi operator.")

    return {
        "jenis": jenis,
        "jenis_confidence": round(type_conf, 4),
        "jenis_top": jenis_top,
        "kondisi": kondisi,
        "kondisi_confidence": round(cond_conf, 4),
        "grade": GRADE_MAP[kondisi],
        "review_flag": type_conf < REVIEW_THRESHOLD or cond_conf < REVIEW_THRESHOLD,
        "catatan": " ".join(notes) if notes else "OK.",
    }
