
# Vorogrid 2D -> 3D

In questa fase il progetto si concentra su:
1. analisi FEM della facciata 2D;
2. costruzione del 3D a partire dalla facciata 2D;
3. nessuna analisi FEM 3D.

## Nota tecnica
La facciata viene analizzata con beam 3D planari per poter assegnare i versori locali di ogni asta:
- `vx` lungo la trave
- `vy = (0,0,1)`
- `vz = vx x vy`
- `geomTransf('Linear', transfTag, *vz)`

## Avvio
```bash
pip install -r requirements.txt
streamlit run app.py
```
