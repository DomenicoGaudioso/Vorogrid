
# Vorogrid Face 2D + OpenSees

Versione corretta dell'applicativo focalizzata SOLO sulla facciata 2D.

## Correzioni principali
- `rigidDiaphragm(1, retained, *nodes)` nel modello 2D della facciata;
- query corretta delle forze degli `elasticBeamColumn`: `eleResponse(..., 'force')`;
- nessun run del modello 3D dalla UI: l'app esegue solo la facciata 2D.

## Avvio
```bash
pip install -r requirements.txt
streamlit run app.py
```
