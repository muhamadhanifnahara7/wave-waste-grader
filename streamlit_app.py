"""
WAVE — Demo grading foto sampah (Streamlit).

Pipeline: foto -> klasifikasi jenis (ResNet50, 6 kelas) -> klasifikasi
kondisi (ResNet50 fine-tune, 3 kelas) -> grade A/B/C.

File yang di-upload ke root Space:
  - app.py, requirements.txt (kode aplikasi)
  - best_grade_model.pth    (model jenis, 6 kelas, akurasi tes 89,5%)
  - best_kondisi_model.pth   (model kondisi, 3 kelas, val 94,1% — kardus)

Kedua file .pth boleh diupload belakangan lewat tab Files di Space.
Selama model belum ada, aplikasi tetap jalan dan menampilkan pesan yang
jelas — tidak crash saat start.

Catatan loader (penting, dari pengalaman di notebook WAVE):
  - best_grade_model.pth BUKAN state dict polos, melainkan dict pembungkus
    (model_state_dict + optimizer + metadata), dan key-nya ber-prefix
    "network." karena model jenis dibungkus class dengan self.network.
    Loader di bawah meng-unwrap dan men-strip prefix itu, lalu melaporkan
    jumlah missing/unexpected — kalau missing mendekati jumlah total
    parameter, berarti bobot tidak termuat dan jangan dipakai.
  - best_kondisi_model.pth bisa berformat state dict polos maupun pembungkus;
    loader yang sama menangani keduanya.
  - Head classifier di checkpoint hasil generate ulang tersimpan sebagai
    Sequential (Dropout + Linear) sehingga key-nya fc.1.weight/fc.1.bias,
    sementara app memakai Linear polos (fc.weight/fc.bias). Loader memetakan
    fc.1.* -> fc.* supaya bobot head terlatih ikut termuat (bukan bobot acak).

Aturan grading:
  - utuh -> A, rusak_ringan -> B, rusak_berat -> C
  - jenis "trash" selalu grade C (aturan pipeline WAVE), tanpa dinilai kondisi
  - confidence < 60% -> flag verifikasi operator
"""

import os

import streamlit as st
import torch
import torch.nn as nn
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
    """Muat checkpoint apa pun formatnya: polos atau pembungkus."""
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
        if (
            key.startswith("fc.")
            and len(key) > 3
            and key[3].isdigit()
        ):
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


# Kedua model opsional saat start, supaya Space tetap jalan sebelum file
# .pth diupload lewat tab Files.
TYPE_MODEL = load_model(TYPE_CKPT, len(TYPE_CLASSES))
COND_MODEL = load_model(COND_CKPT, len(COND_CLASSES))


# ---------------------------------------------------------------- predict
@torch.no_grad()
def predict(image):
    if image is None:
        return {}, "—", "—", "Upload sebuah foto sampah dulu."
    if TYPE_MODEL is None:
        return {}, "—", "—", (
            "Model belum diupload. Upload best_grade_model.pth (dan "
            "best_kondisi_model.pth untuk grading) lewat tab Files di Space "
            "ini, lalu Space akan rebuild otomatis."
        )
    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    type_probs = torch.softmax(TYPE_MODEL(tensor), dim=1)[0]
    type_idx = int(type_probs.argmax())
    jenis = TYPE_CLASSES[type_idx]
    type_conf = float(type_probs[type_idx])
    type_out = {label: round(float(p), 4) for label, p in zip(TYPE_CLASSES, type_probs)}

    notes = []
    if type_conf < REVIEW_THRESHOLD:
        notes.append("Kepercayaan jenis di bawah 60% — perlu verifikasi operator.")

    if jenis == "trash":
        # Aturan pipeline WAVE: sampah campuran hampir selalu grade C.
        return type_out, "— (tidak dinilai)", "C", (
            "Jenis trash selalu grade C sesuai aturan pipeline WAVE."
        )

    if COND_MODEL is None:
        return type_out, "—", "—", (
            "Model kondisi belum diupload — baru jenis yang dinilai. "
            "Upload best_kondisi_model.pth untuk mengaktifkan grading."
        )

    cond_probs = torch.softmax(COND_MODEL(tensor), dim=1)[0]
    cond_idx = int(cond_probs.argmax())
    kondisi = COND_CLASSES[cond_idx]
    cond_conf = float(cond_probs[cond_idx])
    if cond_conf < REVIEW_THRESHOLD:
        notes.append("Kepercayaan kondisi di bawah 60% — perlu verifikasi operator.")
    grade = GRADE_MAP[kondisi]
    kondisi_out = f"{kondisi} ({cond_conf:.0%})"
    return type_out, kondisi_out, grade, " ".join(notes) if notes else "OK."


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="WAVE — Grading Foto Sampah", page_icon="♻️")
st.title("WAVE — Grading Foto Sampah")
st.write(
    "Upload foto sampah untuk mengenali jenis, menilai kondisi, dan menentukan "
    "grade A / B / C."
)
st.info(
    "Model kondisi dilatih pada foto kardus, sehingga hasil untuk bahan lain "
    "bersifat indikatif. Jenis trash selalu grade C."
)

image = st.file_uploader("Foto sampah", type=["jpg", "jpeg", "png"])
if image is not None:
    uploaded_image = Image.open(image).convert("RGB")
    st.image(uploaded_image, caption="Foto yang diunggah", use_container_width=True)
    if st.button("Nilai foto", type="primary"):
        jenis, kondisi, grade, catatan = predict(uploaded_image)
        st.subheader("Hasil")
        st.json(jenis)
        st.write(f"**Kondisi:** {kondisi}")
        st.write(f"**Grade:** {grade}")
        st.write(f"**Catatan:** {catatan}")
