# 1. Base image
FROM python:3.10-slim

# 2. Disable .pyc files & enable stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Work directory
WORKDIR /app

# 4. Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 5. Copy helper scripts + main extractor
COPY download_model.py download_nltk.py extract_headings.py ./

# 6. Download the MiniLM model & NLTK data at build time
RUN python download_model.py \
 && python download_nltk.py

# 7. Ensure input/output dirs exist
RUN mkdir -p input output

# 8. Default entrypoint:
#    For each .pdf in /app/input, run extract_headings.py,
#    writing output to /app/output/{same basename}.json
ENTRYPOINT [ "sh", "-c", "\
  for f in /app/input/*.pdf; do \
    name=$(basename \"$f\" .pdf); \
    python extract_headings.py \"$f\" \
      --model models/all-MiniLM-L6-v2 \
      --out \"/app/output/${name}.json\"; \
  done" ]

