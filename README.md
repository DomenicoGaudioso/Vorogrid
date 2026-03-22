
# Vorogrid 2D -> 3D

Focus attuale:
1. analisi FEM della facciata 2D;
2. costruzione del 3D a partire dalla facciata 2D;
3. nessuna analisi FEM 3D.

## Correzioni di stabilità
- pruning dei componenti non connessi alla base;
- rimozione dei nodi orfani;
- modello 3D planare con vincoli UZ=RX=RY per tutti i nodi non di base.

## Avvio
```bash
pip install -r requirements.txt
streamlit run app.py
```
