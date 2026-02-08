FROM python:3.11-slim

WORKDIR /app

COPY requirements.web.txt .
RUN pip install --no-cache-dir -r requirements.web.txt

COPY . .

RUN mkdir -p data/uploads data/outputs

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
