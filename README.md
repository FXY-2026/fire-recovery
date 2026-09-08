# Recovery-period hydroclimate modulates ecosystem-dependent asynchrony in post-fire carbon–water recovery across Australia

This repository contains the key analysis code and documentation for the study:

**“Recovery-period hydroclimate modulates ecosystem-dependent asynchrony in post-fire carbon–water recovery across Australia”**

*Currently under review for Global Change Biology.*

---

## Data Sources

All datasets used in this study are openly accessible online:

- **MODIS Data (MCD64A1 & MOD09A1):** NASA LP DAAC  
  - [MCD64A1](https://doi.org/10.5067/MODIS/MCD64A1.061)  
  - [MOD09A1](https://doi.org/10.5067/MODIS/MOD09A1.061)

- **TerraClimate:**  
  [https://doi.org/10.1038/sdata.2017.191](https://doi.org/10.1038/sdata.2017.191)

- **BESSv2 (ET & GPP):**  
  [SNU Environment Lab](https://www.environment.snu.ac.kr/bessv2)

- **GOSIF-based GPP:**  
  [Global Ecology Group, UNH](https://globalecology.unh.edu/data/GOSIF-GPP.html)

- **PMLv2 (ET and its components):**  
  [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/projects_pml_evapotranspiration_PML_OUTPUT_PML_V22a)

- **Flux Data (OzFlux):**  
  Obtained from the TERN OzFlux network.

- **uWUE-based ET partitioning:**  
  The external implementation used for flux-site partitioning is available at:  
  [https://github.com/jnelson18/ecosystem-transpiration](https://github.com/jnelson18/ecosystem-transpiration)

---

## Repository Structure

```text
project/
├── recovery_metrics.py              # Extraction of MaxLoss, MTR, and RR
├── RF.py                            # Random Forest classification with spatial-block CV and Optuna tuning
├── SHAP.py                          # SHAP interpretation of fitted Random Forest models
├── mean_e_et_during_recovery.py     # Mean E/ET during the post-fire recovery period
├── et_recovery_status.py            # E and T recovery-status classification at the time of ET recovery
└── README.md
