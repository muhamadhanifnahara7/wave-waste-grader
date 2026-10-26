FROM python:3.11-slim
WORKDIR /app

# CPU-only torch: wheel CUDA terlalu besar untuk Cloud Run.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# File model HARUS ada di folder yang sama saat build, kalau tidak build gagal.
COPY best_grade_model.pth best_kondisi_model.pth ./

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
