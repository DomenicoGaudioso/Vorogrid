
"""app.py
Focus su:
- analisi FEM della facciata 2D;
- costruzione del 3D a valle della facciata 2D;
- nessuna analisi FEM 3D in questa fase.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src import (
    TowerParams,
    generate_face_geometry,
    build_opensees_face_model,
    build_tower_geometry_from_face,
    plotly_face_traces,
    plotly_tower_traces,
    plotly_face_deformed_shape,
    plotly_face_force_map,
    plotly_face_displacement_map,
    export_geometry_json,
)

st.set_page_config(page_title="Vorogrid 2D -> 3D", layout="wide")

st.title("Vorogrid: analisi 2D della facciata + costruzione 3D")
st.caption(
    "In questa fase l'app si concentra su due passaggi: "
    "(1) analisi FEM della facciata 2D e "
    "(2) costruzione del modello geometrico 3D a partire dalla facciata 2D. "
    "L'analisi FEM 3D non viene eseguita."
)

with st.sidebar:
    st.header("Parametri geometrici")
    plan_size = st.number_input("Larghezza facciata [m]", min_value=20.0, max_value=150.0, value=53.0, step=1.0)
    total_height = st.number_input("Altezza totale [m]", min_value=30.0, max_value=800.0, value=351.0, step=1.0)
    n_stories = st.number_input("Numero piani", min_value=5, max_value=200, value=90, step=1)
    core_size = st.number_input("Lato equivalente nucleo [m]", min_value=4.0, max_value=float(plan_size * 0.85), value=25.9, step=0.1)
    n_seeds = st.slider("Numero seed Voronoi", min_value=40, max_value=700, value=260, step=10)
    mode = st.selectbox("Schema densità facciata", ["adaptive", "random", "megaframe", "belts"], index=0)
    random_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=7, step=1)

    st.header("Controllo densità")
    belt_l1 = st.number_input("Belt level 1 [piano]", min_value=1, max_value=int(n_stories - 1), value=min(30, int(n_stories - 1)), step=1)
    belt_l2 = st.number_input("Belt level 2 [piano]", min_value=1, max_value=int(n_stories - 1), value=min(60, int(n_stories - 1)), step=1)
    belt_strength = st.slider("Intensità belt", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
    corner_strength = st.slider("Intensità angoli", min_value=0.0, max_value=5.0, value=1.8, step=0.1)
    base_strength = st.slider("Intensità base", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
    min_edge_len = st.slider("Lunghezza minima aste [m]", min_value=0.2, max_value=4.0, value=0.75, step=0.05)

    st.header("Sezione esoscheletro")
    exo_b = st.number_input("SHS lato esterno [m]", min_value=0.15, max_value=3.0, value=1.10, step=0.01)
    exo_t = st.number_input("SHS spessore [m]", min_value=0.01, max_value=0.40, value=0.09, step=0.01)

    st.header("Carichi modello 2D")
    dead = st.number_input("Carico permanente solaio [kN/m²]", min_value=0.0, max_value=20.0, value=7.0, step=0.1)
    live = st.number_input("Carico variabile solaio [kN/m²]", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
    wind = st.number_input("Azione vento uniforme [kN/m]", min_value=0.0, max_value=1000.0, value=200.0, step=5.0)
    include_vertical_mass = st.checkbox("Includi massa verticale nodale", value=True)
    n_eigen = st.slider("Numero modi propri", min_value=1, max_value=20, value=6, step=1)
    face_load_case = st.selectbox("Caso di carico 2D", ["combined", "lateral", "gravity"], index=0)

    run_btn = st.button("Genera 2D + esegui 2D + costruisci 3D", type="primary")

if 'geometry' not in st.session_state:
    st.session_state.geometry = None
if 'face_analysis' not in st.session_state:
    st.session_state.face_analysis = None
if 'tower_geometry' not in st.session_state:
    st.session_state.tower_geometry = None

if run_btn:
    params = TowerParams(
        plan_size=plan_size,
        total_height=total_height,
        n_stories=int(n_stories),
        core_size=core_size,
        n_seeds=int(n_seeds),
        random_seed=int(random_seed),
        min_edge_len=min_edge_len,
        belt_levels=(int(belt_l1), int(belt_l2)),
        belt_strength=belt_strength,
        corner_strength=corner_strength,
        base_strength=base_strength,
        mode=mode,
        exo_b=exo_b,
        exo_t=exo_t,
        floor_dead_kN_m2=dead,
        floor_live_kN_m2=live,
        wind_line_kN_m=wind,
        include_vertical_mass=include_vertical_mass,
        n_eigen=int(n_eigen),
    )
    with st.spinner("Genero la facciata, eseguo il FEM 2D e costruisco il 3D..."):
        geometry = generate_face_geometry(params)
        st.session_state.geometry = geometry
        try:
            face_analysis = build_opensees_face_model(geometry, load_case=face_load_case, do_eigen=True)
        except Exception as exc:
            face_analysis = {"analysis_ok": False, "error": str(exc)}
        st.session_state.face_analysis = face_analysis
        tower_geometry = build_tower_geometry_from_face(geometry)
        st.session_state.tower_geometry = tower_geometry

geometry = st.session_state.geometry
face_analysis = st.session_state.face_analysis
tower_geometry = st.session_state.tower_geometry

if geometry is None:
    st.info("Imposta i parametri nella barra laterale e premi **Genera 2D + esegui 2D + costruisci 3D**.")
    st.stop()

face = geometry['face']

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodi facciata", len(face['face_nodes']))
c2.metric("Aste facciata", len(face['face_edges']))
c3.metric("Seed Voronoi", len(face['seeds']))
c4.metric("Livelli", len(face['floor_nodes']))

tab1, tab2, tab3, tab4 = st.tabs(["Facciata 2D", "Analisi 2D", "Costruzione 3D", "Esporta"])

with tab1:
    st.plotly_chart(plotly_face_traces(face), use_container_width=True)
    st.write("Questa è la facciata Voronoi 2D che alimenta sia il FEM 2D sia la costruzione del 3D.")

with tab2:
    st.subheader("Analisi FEM della facciata 2D")
    if face_analysis is None:
        st.warning("Nessun risultato FEM 2D disponibile.")
    elif face_analysis.get('analysis_ok', False):
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Caso di carico", str(face_analysis.get('load_case', 'combined')))
        cc2.metric("Ux max [m]", f"{face_analysis.get('top_ux', 0.0):.4e}")
        cc3.metric("|u| max [m]", f"{face_analysis.get('top_umag', 0.0):.4e}")
        cc4.metric("Aste analizzate", len(face_analysis.get('edges', [])))

        copt1, copt2, copt3 = st.columns([1, 1, 1])
        disp_component = copt1.selectbox("Mappa spostamenti nodali", ["ux", "uz", "umag"], index=0)
        force_quantity = copt2.selectbox("Mappa grandezza di elemento", ["N", "V", "M"], index=2)
        deformed_force = copt3.checkbox("Mostra mappa forze sulla deformata", value=False)
        scale = st.slider("Fattore scala deformata", min_value=1.0, max_value=200.0, value=25.0, step=1.0)

        fg1, fg2 = st.columns(2)
        with fg1:
            st.plotly_chart(plotly_face_deformed_shape(face_analysis, scale=scale), use_container_width=True)
        with fg2:
            st.plotly_chart(plotly_face_displacement_map(face_analysis, component=disp_component), use_container_width=True)

        st.plotly_chart(plotly_face_force_map(face_analysis, quantity=force_quantity, deformed=deformed_force, scale=scale), use_container_width=True)

        periods = face_analysis.get('periods_s', [])
        if periods:
            st.subheader("Modi propri 2D")
            st.dataframe(pd.DataFrame({"Modo": list(range(1, len(periods) + 1)), "Periodo [s]": periods}), use_container_width=True)

        axes = pd.DataFrame(face_analysis.get('elem_local_axes', []))
        if not axes.empty:
            st.subheader("Versori locali dei primi elementi")
            st.dataframe(axes.head(20), use_container_width=True)
    else:
        st.error("La facciata è stata generata, ma OpenSeesPy non ha completato l'analisi 2D.")
        if isinstance(face_analysis, dict) and face_analysis.get('error'):
            st.code(face_analysis['error'])
        st.info("Installa `openseespy` per eseguire il run del modello 2D nell'app.")

with tab3:
    st.subheader("Costruzione del 3D a partire dalla facciata 2D")
    if tower_geometry is not None:
        st.plotly_chart(plotly_tower_traces(tower_geometry), use_container_width=True)
        st.write("Qui viene costruita la geometria 3D della torre replicando la facciata 2D sulle quattro facce e aggiungendo un nucleo geometrico equivalente. In questa fase non viene eseguita l'analisi 3D.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Nodi esoscheletro 3D", len(tower_geometry['tower']['nodes']))
        m2.metric("Aste esoscheletro 3D", len(tower_geometry['tower']['edges']))
        m3.metric("Nodi nucleo 3D", len(tower_geometry['core']['nodes']))
        m4.metric("Shell nucleo 3D", len(tower_geometry['core']['shells']))
    else:
        st.warning("Nessuna geometria 3D disponibile.")

with tab4:
    st.download_button("Scarica geometria facciata JSON", data=export_geometry_json(geometry), file_name="vorogrid_face_geometry.json", mime="application/json")
    if tower_geometry is not None:
        st.download_button("Scarica geometria torre 3D JSON", data=export_geometry_json(tower_geometry), file_name="vorogrid_tower_geometry.json", mime="application/json")
    st.code("streamlit run app.py", language="bash")
