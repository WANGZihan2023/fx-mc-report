# Streamlit FX report — WeasyPrint-capable image (Railway / Render / Fly)
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # WeasyPrint / fontconfig
    FX_PDF_ENGINE=weasyprint

# System libs for WeasyPrint (pango / cairo / gdk-pixbuf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-dejavu-core \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App + bundled Stage-1 calib (fx_report/data/calibrated/). output/ is dockerignored.
COPY . .

# Railway / Render inject PORT; default 8501 for local docker run
ENV PORT=8501
EXPOSE 8501

# Bind 0.0.0.0 so the platform proxy can reach Streamlit.
# Disable XSRF/CORS checks — required behind Railway HTTPS reverse proxy
# (otherwise the UI shell loads but stays blank).
CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
