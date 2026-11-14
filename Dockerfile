FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY nba_platform ./nba_platform
RUN pip install --no-cache-dir .
# Cloud Run Job / Fly machine entrypoints:
#   daily: nba-platform daily --db $DB_PATH      (once, ~09:00 ET)
#   tick:  nba-platform tick  --db $DB_PATH      (every 5 min, 11:00-01:00 ET on game days)
ENTRYPOINT ["nba-platform"]
