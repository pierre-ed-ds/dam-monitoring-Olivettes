# Olivettes - Water Level Analysis and Forecasting

## Description
Olivettes is a desktop Python application for analyzing and forecasting water levels of a reservoir (Olivettes Dam). It allows you to:

- Load water level data from CSV files.
- Visualize levels on interactive charts.
- Define critical thresholds and operational zones.
- Simulate level forecasts based on outflow and evaporation.
- Manually adjust forecasted values.
- Export charts as PNG, PDF, or JPEG.

The application uses **Tkinter** with the **Flatly** theme via **ttkbootstrap**, as well as **matplotlib**, **pandas**, **scipy**, and **mplcursors** for interactive plots.

---

## Main Features

![Olivettes Screenshot](Impr_ecr.png)

### CSV Data Loading
- The CSV must contain a **date** column (or datetime index) and a **COTE** column (water level in mNGF).  

### Interactive Visualization
- Matplotlib charts embedded in the Tkinter interface.
- Navigation via toolbar (zoom, pan, save).
- Interactive tooltips showing date, water level, total volume, and usable volume.

### Level Forecasting
- Based on historical data and monthly assumptions:
  - Water discharge (m³/s)
  - Evaporation (mm/day)
- Charts display colored areas corresponding to critical thresholds.

### Manual Adjustments
- Allows adjusting the level for a specific date to refine the forecast.

### Chart Export
- Export charts as PNG, PDF, or JPEG.
- Option to remove interactive tooltips before exporting.

---

## Installation

### Requirements
- Python 3.10+
- Install dependencies with:
```bash
pip install -r requirements.txt
```
Required HSV file: HSV_32.txt (contains the relations COTE ↔ VOLUME ↔ SURFACE)

### PyInstaller Option
```
pyinstaller --onefile --noconsole app_olivettes.py \
--add-data "HSV_32.txt;." \
--add-data "<path_to_python>/Lib/site-packages/matplotlib/mpl-data;matplotlib/mpl-data" \
--hidden-import=matplotlib.backends.backend_pdf
```
Last line is required to download the plots as pdf.


