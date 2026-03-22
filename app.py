
"""app.py
Pipeline:
1. Analisi FEM della facciata 2D (OpenSees ndm=2, ndf=3).
2. Costruzione del modello geometrico 3D a partire dalla facciata 2D.
3. Analisi FEM 3D della torre (elasticBeamColumn + ShellMITC4 + rigidDiaphragm).

Riferimento: Laccone et al. – "VoroGrid: Geometrically-informed Voronoi
Tessellation for Structural Design of Tall Buildings" (CAD/CAE Integration).
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src import (
    TowerParams,
    generate_face_geometry,
    build_opensees_face_model,
    build_tower_geometry_from_face,
    build_opensees_tower_model,
    plotly_face_traces,
    plotly_tower_traces,
    plotly_face_deformed_shape,
    plotly_face_force_map,
    plotly_face_displacement_map,
    plotly_face_drift_profile,
    plotly_tower_deformed_traces,
    plotly_tower_force_map,
    plotly_tower_drift_profile,
    export_geometry_json,
)

st.set_page_config(page_title="VoroGrid – 2D + 3D FEM", layout="wide")

st.title("VoroGrid: analisi FEM 2D + 3D della torre Voronoi")
st.caption(
    "2D: telaio piano (ndm=2, ndf=3) · "
    "3D: elasticBeamColumn + ShellMITC4 + rigidDiaphragm · "
    "— Rif.: *Laccone et al., VoroGrid, CAD/CAE Integration*"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Geometria torre")
    plan_size    = st.number_input("Larghezza facciata [m]", min_value=20.0,  max_value=150.0, value=53.0,  step=1.0)
    total_height = st.number_input("Altezza totale [m]",     min_value=30.0,  max_value=800.0, value=351.0, step=1.0)
    n_stories    = st.number_input("Numero piani",           min_value=5,     max_value=200,   value=90,    step=1)
    core_size    = st.number_input("Lato equivalente nucleo [m]",
                                    min_value=4.0, max_value=float(plan_size * 0.85),
                                    value=min(25.9, plan_size * 0.85 - 0.1), step=0.1)

    st.header("Maglia Voronoi")
    n_seeds     = st.slider("Numero seed",       min_value=40,  max_value=700,   value=260, step=10)
    mode        = st.selectbox("Schema densità", ["adaptive", "random", "megaframe", "belts"], index=0)
    random_seed = st.number_input("Random seed", min_value=0,   max_value=99999, value=7,   step=1)

    st.header("Densità adattiva")
    belt_l1         = st.number_input("Belt level 1 [piano]",  min_value=1, max_value=int(n_stories-1), value=min(30, int(n_stories-1)), step=1)
    belt_l2         = st.number_input("Belt level 2 [piano]",  min_value=1, max_value=int(n_stories-1), value=min(60, int(n_stories-1)), step=1)
    belt_strength   = st.slider("Intensità belt",   min_value=0.0, max_value=5.0, value=2.0, step=0.1)
    corner_strength = st.slider("Intensità angoli", min_value=0.0, max_value=5.0, value=1.8, step=0.1)
    base_strength   = st.slider("Intensità base",   min_value=0.0, max_value=5.0, value=2.5, step=0.1)
    min_edge_len    = st.slider("Lunghezza minima aste [m]", min_value=0.2, max_value=4.0, value=0.75, step=0.05)

    st.header("Sezione SHS esoscheletro")
    exo_b = st.number_input("Lato esterno b [m]",  min_value=0.15, max_value=3.0,  value=1.10, step=0.01)
    exo_t = st.number_input("Spessore t [m]",       min_value=0.01, max_value=0.40, value=0.09, step=0.01)

    st.header("Nucleo in c.a.")
    core_t       = st.number_input("Spessore pareti [m]",    min_value=0.10, max_value=2.0,  value=0.50,  step=0.05)
    concrete_E   = st.number_input("E calcestruzzo [GPa]",   min_value=20.0, max_value=50.0, value=30.0,  step=1.0)
    concrete_nu  = st.number_input("ν calcestruzzo",         min_value=0.10, max_value=0.30, value=0.20,  step=0.01)
    concrete_rho = st.number_input("ρ calcestruzzo [kg/m³]", min_value=2000, max_value=3000, value=2500,  step=50)

    st.header("Carichi e dinamica")
    dead    = st.number_input("Carico perm. solaio [kN/m²]", min_value=0.0, max_value=20.0,   value=7.0,   step=0.1)
    live    = st.number_input("Carico var. solaio [kN/m²]",  min_value=0.0, max_value=20.0,   value=3.0,   step=0.1)
    wind    = st.number_input("Vento uniforme [kN/m]",        min_value=0.0, max_value=1000.0, value=200.0, step=5.0)
    include_vertical_mass = st.checkbox("Includi massa verticale nodale", value=True)
    n_eigen = st.slider("Modi propri", min_value=1, max_value=20, value=6, step=1)
    face_load_case  = st.selectbox("Caso carico 2D",  ["combined", "lateral", "gravity"], index=0)
    tower_load_case = st.selectbox("Caso carico 3D",  ["lateral", "combined", "gravity"], index=0)

    run2d_btn  = st.button("Genera + Analisi 2D + Costruzione 3D", type="primary")
    run3d_btn  = st.button("Esegui Analisi FEM 3D", type="secondary")

# ── Session state ─────────────────────────────────────────────────────────────
for key in ('geometry', 'face_analysis', 'tower_geometry', 'tower_analysis'):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Run 2D ────────────────────────────────────────────────────────────────────
if run2d_btn:
    params = TowerParams(
        plan_size=plan_size, total_height=total_height, n_stories=int(n_stories),
        core_size=core_size, n_seeds=int(n_seeds), random_seed=int(random_seed),
        min_edge_len=min_edge_len, belt_levels=(int(belt_l1), int(belt_l2)),
        belt_strength=belt_strength, corner_strength=corner_strength,
        base_strength=base_strength, mode=mode, exo_b=exo_b, exo_t=exo_t,
        floor_dead_kN_m2=dead, floor_live_kN_m2=live, wind_line_kN_m=wind,
        include_vertical_mass=include_vertical_mass, n_eigen=int(n_eigen),
        concrete_E=float(concrete_E) * 1e9, concrete_nu=float(concrete_nu),
        concrete_rho=float(concrete_rho), core_t=float(core_t),
    )
    with st.spinner("Generazione Voronoi + analisi FEM 2D + costruzione 3D…"):
        geometry = generate_face_geometry(params)
        st.session_state.geometry = geometry
        st.session_state.tower_analysis = None  # reset 3D
        try:
            face_analysis = build_opensees_face_model(geometry, load_case=face_load_case, do_eigen=True)
        except Exception as exc:
            face_analysis = {"analysis_ok": False, "error": str(exc)}
        st.session_state.face_analysis = face_analysis
        tower_geometry = build_tower_geometry_from_face(geometry)
        st.session_state.tower_geometry = tower_geometry

# ── Run 3D ────────────────────────────────────────────────────────────────────
if run3d_btn:
    geometry       = st.session_state.geometry
    tower_geometry = st.session_state.tower_geometry
    if geometry is None or tower_geometry is None:
        st.sidebar.error("Esegui prima l'analisi 2D per generare la geometria 3D.")
    else:
        # Aggiorna i parametri c.a. nel geometry['params'] prima dell'analisi 3D
        geometry['params'].update({
            'concrete_E':   float(concrete_E) * 1e9,
            'concrete_nu':  float(concrete_nu),
            'concrete_rho': float(concrete_rho),
            'core_t':       float(core_t),
            'n_eigen':      int(n_eigen),
        })
        with st.spinner("Analisi FEM 3D (elasticBeamColumn + ShellMITC4 + rigidDiaphragm)…"):
            try:
                tower_analysis = build_opensees_tower_model(
                    tower_geometry, geometry, load_case=tower_load_case, do_eigen=True
                )
            except Exception as exc:
                tower_analysis = {"analysis_ok": False, "error": str(exc)}
        st.session_state.tower_analysis = tower_analysis

geometry       = st.session_state.geometry
face_analysis  = st.session_state.face_analysis
tower_geometry = st.session_state.tower_geometry
tower_analysis = st.session_state.tower_analysis

if geometry is None:
    st.info("Imposta i parametri nella barra laterale e premi **Genera + Analisi 2D + Costruzione 3D**.")
    st.stop()

face = geometry['face']

# ── Metriche geometria ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodi facciata",    len(face['face_nodes']))
c2.metric("Aste facciata",    len(face['face_edges']))
c3.metric("Seed Voronoi",     len(face['seeds']))
c4.metric("Livelli di piano", len(face['floor_nodes']))

prune = face.get('prune_stats', {})
if prune:
    st.info(
        f"Pulizia topologica: componenti trovati={prune.get('components_total',0)}, "
        f"tenuti={prune.get('components_kept',0)}, "
        f"nodi rimossi={prune.get('nodes_removed',0)}, "
        f"aste rimosse={prune.get('edges_removed',0)}."
    )

# ── Tab ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Facciata 2D",
    "Analisi FEM 2D",
    "Deriva 2D",
    "Geometria 3D",
    "Analisi FEM 3D",
    "Deriva 3D",
    "Esporta",
])

# ── Tab 1 – Geometria facciata ─────────────────────────────────────────────────
with tab1:
    st.plotly_chart(plotly_face_traces(face), use_container_width=True)
    story_h = total_height / n_stories
    st.caption(
        f"Facciata {plan_size:.0f} m × {total_height:.0f} m — "
        f"interpiano {story_h:.2f} m — schema densità: **{mode}**"
    )

# ── Tab 2 – Analisi FEM 2D ────────────────────────────────────────────────────
with tab2:
    st.subheader("Analisi FEM della facciata 2D (OpenSees ndm=2, ndf=3)")
    if face_analysis is None:
        st.warning("Nessun risultato FEM 2D disponibile.")
    elif face_analysis.get('analysis_ok', False):
        top_ux    = face_analysis.get('top_ux', 0.0)
        top_umag  = face_analysis.get('top_umag', 0.0)
        max_drift = face_analysis.get('max_drift_ratio', 0.0)
        h_lim     = total_height / 500.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Caso di carico",  str(face_analysis.get('load_case', '-')))
        m2.metric("ux,max [m]",      f"{top_ux:.4e}")
        m3.metric("|u|max [m]",      f"{top_umag:.4e}")
        m4.metric("Drift max [−]",   f"{max_drift:.4e}")
        m5.metric("H/500 [m]",       f"{h_lim:.4e}",
                  delta=f"{'OK' if top_ux <= h_lim else 'SUPERA'}",
                  delta_color="normal" if top_ux <= h_lim else "inverse")

        st.caption(
            f"Modello 2D: {face_analysis.get('n_active_nodes', '?')} nodi attivi "
            f"({len(face_analysis.get('base_node_tags',[]))} incastrati); "
            f"{face_analysis.get('n_active_elements', '?')} elementi. "
            f"Nodi geometria: {len(face['face_nodes'])}."
        )

        copt1, copt2, copt3 = st.columns(3)
        disp_comp  = copt1.selectbox("Mappa spostamenti",      ["ux", "uz", "umag"], index=0)
        force_qty  = copt2.selectbox("Mappa forze di elemento",["N", "V", "M"],      index=2)
        def_force  = copt3.checkbox("Forze sulla deformata", value=False)
        scale_2d   = st.slider("Scala deformata 2D", min_value=1.0, max_value=500.0, value=25.0, step=1.0)

        fg1, fg2 = st.columns(2)
        with fg1:
            st.plotly_chart(plotly_face_deformed_shape(face_analysis, scale=scale_2d), use_container_width=True)
        with fg2:
            st.plotly_chart(plotly_face_displacement_map(face_analysis, component=disp_comp), use_container_width=True)
        st.plotly_chart(plotly_face_force_map(face_analysis, quantity=force_qty, deformed=def_force, scale=scale_2d), use_container_width=True)

        periods = face_analysis.get('periods_s', [])
        if periods:
            st.subheader("Modi propri 2D")
            st.dataframe(pd.DataFrame({
                "Modo": list(range(1, len(periods)+1)),
                "T [s]":  [f"{p:.4f}" for p in periods],
                "f [Hz]": [f"{1/p:.4f}" if p > 0 else "–" for p in periods],
            }), use_container_width=True)
    else:
        st.error("OpenSeesPy non ha completato l'analisi 2D.")
        if isinstance(face_analysis, dict) and face_analysis.get('error'):
            st.code(face_analysis['error'])
        st.info("Verifica l'installazione di `openseespy` e riduci la lunghezza minima delle aste se necessario.")

# ── Tab 3 – Deriva 2D ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Profilo spostamenti e drift inter-piano (2D)")
    if face_analysis and face_analysis.get('analysis_ok', False):
        sh = total_height / n_stories
        st.plotly_chart(plotly_face_drift_profile(face_analysis, n_stories=int(n_stories), story_height=sh), use_container_width=True)
        st.caption("La linea tratteggiata arancione indica il limite h/500 per interpiano.")
    else:
        st.warning("Esegui prima l'analisi FEM 2D.")

# ── Tab 4 – Geometria 3D ──────────────────────────────────────────────────────
with tab4:
    st.subheader("Geometria 3D – esoscheletro + nucleo")
    if tower_geometry is not None:
        st.plotly_chart(plotly_tower_traces(tower_geometry), use_container_width=True)
        st.caption(
            "Facciata 2D replicata su 4 facce + nucleo rettangolare equivalente. "
            "Analisi FEM 3D separata (bottone sidebar)."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Nodi esoscheletro 3D", len(tower_geometry['tower']['nodes']))
        m2.metric("Aste esoscheletro 3D", len(tower_geometry['tower']['edges']))
        m3.metric("Nodi nucleo 3D",       len(tower_geometry['core']['nodes']))
        m4.metric("Shell nucleo 3D",      len(tower_geometry['core']['shells']))
    else:
        st.warning("Nessuna geometria 3D disponibile.")

# ── Tab 5 – Analisi FEM 3D ────────────────────────────────────────────────────
with tab5:
    st.subheader("Analisi FEM 3D (elasticBeamColumn + ShellMITC4 + rigidDiaphragm)")

    if tower_analysis is None:
        st.info(
            "Genera prima la geometria 2D/3D con il primo bottone, "
            "poi premi **Esegui Analisi FEM 3D** nella barra laterale."
        )
    elif tower_analysis.get('analysis_ok', False):
        top_ux_3d    = tower_analysis.get('top_ux', 0.0)
        top_umag_3d  = tower_analysis.get('top_umag', 0.0)
        drift_3d     = tower_analysis.get('max_drift_ratio', 0.0)
        h_lim        = total_height / 500.0

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Caso carico",       str(tower_analysis.get('load_case', '-')))
        m2.metric("ux,top [m]",        f"{top_ux_3d:.4e}")
        m3.metric("|u|top [m]",        f"{top_umag_3d:.4e}")
        m4.metric("Drift max 3D [−]",  f"{drift_3d:.4e}")
        m5.metric("H/500 [m]",         f"{h_lim:.4e}",
                  delta=f"{'OK' if top_ux_3d <= h_lim else 'SUPERA'}",
                  delta_color="normal" if top_ux_3d <= h_lim else "inverse")
        m6.metric("Shell nucleo",      str(tower_analysis.get('n_shell_elements', '?')))

        st.caption(
            f"Nodi esoscheletro attivi: {tower_analysis.get('n_exo_nodes','?')} · "
            f"Elementi trave: {tower_analysis.get('n_exo_elements','?')} · "
            f"Shell: {tower_analysis.get('n_shell_elements','?')} · "
            f"Diaframmi rigidi: uno per piano (perpDirn=3)."
        )

        scale_3d = st.slider("Scala deformata 3D", min_value=1.0, max_value=500.0, value=10.0, step=1.0)
        force_3d = st.selectbox("Mappa forze esoscheletro 3D", ["N", "V", "M"], index=0)

        tab5a, tab5b = st.tabs(["Deformata 3D", "Mappa forze 3D"])
        with tab5a:
            st.plotly_chart(plotly_tower_deformed_traces(tower_analysis, scale=scale_3d), use_container_width=True)
        with tab5b:
            st.plotly_chart(plotly_tower_force_map(tower_analysis, quantity=force_3d), use_container_width=True)

        periods_3d = tower_analysis.get('periods_s', [])
        if periods_3d:
            st.subheader("Modi propri 3D")
            st.dataframe(pd.DataFrame({
                "Modo": list(range(1, len(periods_3d)+1)),
                "T [s]":  [f"{p:.4f}" for p in periods_3d],
                "f [Hz]": [f"{1/p:.4f}" if p > 0 else "–" for p in periods_3d],
            }), use_container_width=True)

    else:
        st.error("L'analisi FEM 3D non è riuscita.")
        if isinstance(tower_analysis, dict) and tower_analysis.get('error'):
            st.code(tower_analysis['error'])
        st.info(
            "Possibili cause: elementi troppo corti (aumenta lunghezza minima aste), "
            "spessore nucleo insufficiente, o `openseespy` non installato."
        )

# ── Tab 6 – Deriva 3D ─────────────────────────────────────────────────────────
with tab6:
    st.subheader("Profilo spostamenti e drift inter-piano (3D)")
    if tower_analysis and tower_analysis.get('analysis_ok', False):
        sh3d = total_height / n_stories
        st.plotly_chart(plotly_tower_drift_profile(tower_analysis, story_height=sh3d), use_container_width=True)
        st.caption("Spostamenti medi degli nodi esoscheletro per livello, direzione X (vento).")
    else:
        st.warning("Esegui prima l'analisi FEM 3D.")

# ── Tab 7 – Export ────────────────────────────────────────────────────────────
with tab7:
    st.subheader("Esporta geometria")
    st.download_button(
        "Scarica geometria facciata (JSON)",
        data=export_geometry_json(geometry),
        file_name="vorogrid_face_geometry.json",
        mime="application/json",
    )
    if tower_geometry is not None:
        st.download_button(
            "Scarica geometria torre 3D (JSON)",
            data=export_geometry_json(tower_geometry),
            file_name="vorogrid_tower_geometry.json",
            mime="application/json",
        )
    st.divider()
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
