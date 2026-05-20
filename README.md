# Berechnungen von Distanzmatrizen für verschiedene Feature-Gruppen
Dieses Repository enthält den Code zur Berechnung paarweiser Kundendistanzmatrizen für PFERD TOOLS auf Basis von drei Feature-Gruppen: NAICS-Branchencodes, HS-Handelscodes und Applikations-Codes.
Der Workflow gliedert sich in drei domänenspezifische Distanz-Engines, die unabhängig voneinander ausgeführt werden können:

NAICS-Distanzen — Ziffernbasierte Metriken (Hamming, gewichtetes Hamming, LCP, Block-Gewichtung) auf Primary- und Secondary-NAICS-Codes mit konfigurierbarem Gating und Alpha-Penalty

HS-Distanzen — Zwei Berechnungswege: eine Set-Engine (Jaccard, Dice, Overlap inkl. IDF-gewichteter Varianten) und eine Punkt-Engine (Cartesian Mean, Chamfer, kNN) auf normalisierten HS6-Ziffern-Arrays

Applikations-Distanzen — Gleiche Zwei-Engine-Architektur wie HS, angewandt auf den Applikationsteil der AM-Codes mit optionalem produktspezifischen Filter

Zusätzlich enthält das Repository eine Grid-Search-Schicht (parameter_grid), die systematisch Distanzmatrizen über alle Parameterkombinationen erzeugt, als .pkl persistiert und in einer Registry dokumentiert. Ein Konvertierungsmodul (convert_to_long) überführt quadratische Matrizen ins Long-Format für die nachgelagerte Analyse.
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
