# Hugging Face Spaces (Docker SDK) image for the Streamlit dashboard.
# Python 3.13 matches the version the pinned requirements were tested on.
FROM python:3.13-slim

# Spaces runs the container as a non-root user with uid 1000.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
USER user
RUN pip install --no-cache-dir -r requirements.txt

# data/processed is committed, so the dashboard starts without rebuilding.
# (If it were missing, app/common.py rebuilds it on first start.)
COPY --chown=user . .

# GEMINI_API_KEY is supplied at runtime as a Space secret (never baked into the image).
EXPOSE 7860
CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
