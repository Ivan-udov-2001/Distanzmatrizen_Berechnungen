# Distanzmatrizen_Berechnungen
Berechnung von Kundendistanzen für PFERD TOOLS auf Basis von NAICS-, HS- und Applikations-/Materialcodes.
## Projektstruktur

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── data/
│   └── df_final_master_pseudo.pkl
│
├── src/
│   ├── __init__.py
│   ├── application.py
│   ├── convert_to_long.py
│   ├── core_metrics.py
│   ├── hs.py
│   ├── naics.py
│   ├── parameter_grid.py
│   ├── paths.py
│   ├── set_metrics.py
│   │
│   └── run_parameter_grid/
│       ├── __init__.py
│       ├── run_grid_search_APP.py
│       ├── run_grid_search_HS.py
│       └── run_grid_search_NAICS.py
│
├── tests/
│   ├── __init__.py
│   ├── integration/
│   │   └── test_integration.py
│   └── unit/
│       ├── test_application.py
│       ├── test_hs.py
│       ├── test_naics.py
│       ├── test_parameter_grid.py
│       ├── test_set_metrics.py
│       └── test_special_cases.py
│
├── scripts/
│   └── pipeline.ipynb
│
├── grid_outputs/
│   ├── APP/
│   │   └── APP_long/
│   ├── HS/
│   │   └── HS_long/
│   └── NAICS/
│       ├── NAICS_long/
│       └── NAICS_square.zip
│
├── Histogramme/
│   ├── APP/
│   ├── HS/
│   └── NAICS/
│
└── Heatmaps/
    ├── APP/
    ├── HS/
    └── NAICS/
