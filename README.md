# Gafas IA API

## Desplegar
1. Render.com → New Web Service → tu repo
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## Uso
POST `/predict` con imagen → JSON con detecciones