import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import sys, os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends import backend_pdf
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import matplotlib.dates as mdates
import mplcursors

def on_closing():
    plt.close('all')   # ferme toutes les figures matplotlib
    root.destroy()
    sys.exit()


# === Fonctions ===
def load_csv():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        try:
            global data_raw  # on garde ici le df brut
            df = pd.read_csv(filepath, sep=";", decimal=",")
            # renommage éventuel de la colonne
            if "Cote du plan d'eau (mNGF)" in df.columns:
                df = df.rename(columns={"Cote du plan d'eau (mNGF)": "COTE"})

            data_raw = df  # stocke la version brute
            messagebox.showinfo("Succès", f"Données chargées depuis {filepath}")
            return data_raw

        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            return None


def plot_cote(frame, date_col="Date", value_col="COTE", start_date="01/01/2007", df=None):
    """
    Si df est None, utilise la variable globale `data_raw` (données brutes).
    Met à jour globals()['data'] avec la version nettoyée / filtrée.
    Dessine le graphe dans `frame`.
    """
    global data, data_raw
    try:
        # récupérer le df (global si non fourni)
        if df is None:
            if 'data_raw' not in globals():
                messagebox.showerror("Aucune donnée", "La variable globale 'data_raw' n'existe pas.")
                return
            df = globals()['data_raw']

        # normaliser en DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df2 = df.copy()

        # --- détection colonne date ou index datetime ---
        if date_col not in df2.columns:
            if isinstance(df2.index, pd.DatetimeIndex):
                pass
            else:
                candidates = [c for c in df2.columns if any(k in c.lower() for k in ("date", "time"))]
                if candidates:
                    used_date_col = candidates[0]
                    df2[used_date_col] = pd.to_datetime(df2[used_date_col], dayfirst=True, errors='coerce')
                    df2 = df2.dropna(subset=[used_date_col]).set_index(used_date_col).sort_index()
                else:
                    idx_parsed = pd.to_datetime(df2.index, dayfirst=True, errors='coerce')
                    if idx_parsed.notna().any():
                        df2.index = idx_parsed
                        df2 = df2.dropna(axis=0, how='all')
                    else:
                        messagebox.showerror(
                            "Colonne date introuvable",
                            f"Impossible de trouver une colonne date.\nColonnes disponibles: {list(df2.columns)}"
                        )
                        return
        else:
            df2[date_col] = pd.to_datetime(df2[date_col], dayfirst=True, errors='coerce')
            df2 = df2.dropna(subset=[date_col]).set_index(date_col).sort_index()

        # --- parsing start_date ---
        try:
            sd = pd.to_datetime(start_date, dayfirst=True)
        except Exception:
            sd = df2.index.min()

        df2 = df2[df2.index >= sd]

        # --- colonne valeur ---
        if value_col not in df2.columns:
            if df2.shape[1] == 1:
                value_col = df2.columns[0]
            else:
                messagebox.showerror(
                    "Colonne valeur introuvable",
                    f"'{value_col}' non trouvée. Colonnes disponibles: {list(df2.columns)}"
                )
                return

        # nettoyage valeur (virgule -> point) et conversion numérique
        df2[value_col] = df2[value_col].astype(str).str.replace(",", ".", regex=False)
        df2[value_col] = pd.to_numeric(df2[value_col], errors='coerce')

        plot_data = df2[value_col].dropna()
        if plot_data.empty:
            messagebox.showwarning("Données vides", "Aucune date entrée")
            globals()['data'] = df2
            return

        # --- tracé ---
        fig, ax = plt.subplots()
        plot_data.plot(ax=ax, grid=True, title=f"Cote depuis {sd.date()}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cote (mNGF)")

        # pour mieux gérer l'espace
        fig.tight_layout()

        # nettoyer l'ancien canvas
        for widget in frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- mise à jour de la globale avec la version nettoyée ---
        globals()['data'] = df2  # data = filtrée
        # data_raw reste intact

    except Exception as exc:
        messagebox.showerror("Erreur", "Une erreur est survenue — voir la console pour la trace.")


def make_hsv_converters(filepath):
    df = pd.read_csv(filepath, sep=";")
    cote_to_vol = interp1d(df["COTE"], df["VOLUME"], kind="linear", fill_value="extrapolate")
    vol_to_cote = interp1d(df["VOLUME"], df["COTE"], kind="linear", fill_value="extrapolate")
    cote_to_surf = interp1d(df["COTE"], df["SURFACE"], kind="linear", fill_value="extrapolate")
    return cote_to_vol, vol_to_cote, cote_to_surf


def forecast_volume(data, cote_to_vol, vol_to_cote, cote_to_surf,
                    debit_sortant_dict, evap_dict, frame,
                    thresholds, colors, alpha=0.2, value_col="COTE", zoom=False,
                    adjustments=None):
    from tkinter import messagebox

    last_date = data.index.max()
    if not hasattr(last_date, 'year'):
        messagebox.showerror("Erreur", "L'index n'est pas en datetime. Tracez d'abord la cote.")
        return

    end_year = pd.Timestamp(year=last_date.year, month=12, day=31)
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), end_year, freq="D")

    last_cote = data[value_col].iloc[-1]
    volume = float(cote_to_vol(last_cote))
    surface = float(cote_to_surf(last_cote))

    results = []
    for date in future_dates:
        mois = date.month
        Qs = debit_sortant_dict.get(mois, 0)
        evap = evap_dict.get(mois, 0)
        perte = Qs * 86400 + evap * surface * 10000 / 1000
        volume = volume - perte
        cote = float(vol_to_cote(volume))
        surface = float(cote_to_surf(cote))

        if adjustments is not None:
            adj_value = adjustments.get(pd.Timestamp(date), 0)
            cote += adj_value
            volume = float(cote_to_vol(cote))
            surface = float(cote_to_surf(cote))

        results.append((date, cote))

    forecast_df = pd.DataFrame(results, columns=["DATE", "COTE"]).set_index("DATE")

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.subplots_adjust(bottom=0.2,right=0.85)

    ymin= 150 #ymin = min(forecast_df[value_col].min(),150)
    ymax = 165 #max(forecast_df[value_col].max(), 163)
    ax.axhspan(ymin, thresholds[0], color='red', alpha=alpha)
    for i in range(1, len(thresholds)):
        ax.axhspan(thresholds[i-1], thresholds[i], color=colors[i], alpha=alpha)
    ax.axhspan(thresholds[-1], ymax, color=colors[-1], alpha=alpha)

    # Plot Observé et Prévision
    if zoom:
        start_year = pd.Timestamp(year=last_date.year, month=1, day=1)
        data_to_plot = data.loc[start_year:]
        # mois en français 
        mois_fr = ["", "Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"] 
        months = pd.date_range(start=start_year, end=end_year, freq='MS') 
        month_labels = [mois_fr[m.month] for m in months] 
        month_positions = months
        ax.plot(data_to_plot.index, data_to_plot[value_col], label="Observé", linestyle="-", linewidth=2)
        ax.plot(forecast_df.index, forecast_df[value_col], label="Prévision", linestyle="--")
        ax.set_xticks(month_positions) 
        ax.set_xticklabels(month_labels)
    else:
        ax.plot(data.index, data[value_col], label="Observé", linewidth=2)
        ax.plot(forecast_df.index, forecast_df[value_col], linestyle="--", label="Prévision")
    
    ax.axhline(y=152, color='red', linestyle='-', linewidth=1.5)
    ax.axhline(y=163, color='black', linestyle='-', linewidth=1.5)

    ax.set_yticks(range(150, 166))  # graduations chaque 1 mètre
    #ax.set_yticklabels([" "] * 2+["152"] + [" "] * 10 + ["163"]+[" "] )  # seuls 152 et 163 sont visibles
    ax.set_ylabel("Cote (mNGF)")

    # Axe secondaire (droite) pour le volume
    ax2 = ax.twinx()

    # Conversion linéaire cote -> volume
    # 152 mNGF -> 0 m3
    # 163 mNGF -> 3.22e6 m3
    ax2.set_ylim(ax.get_ylim())

    # Synchroniser les graduations avec celles de l'axe principal
    ax2.set_yticks(range(150, 165))
    ax2.set_yticklabels([" "] * 2+["0 m³"] + [" "] * 3+["0.92 Mm³"]+ [" "] * 3 +["2.11 Mm³"]+[" "] * 2 + ["3.22 Mm³"]+[" "] )
    ax2.set_ylabel("Volume exploitable")
    if zoom:
        ax.set_xlabel("Mois")
    else:
        ax.set_xlabel("Années")

    # --- Lignes (Observé / Prévision) ---
    line_handles, line_labels = ax.get_legend_handles_labels()  # récupère les lignes
    rename_dict = {"Observé": "Cote réelle du plan d'eau",
                   "Prévision": "Projection de la cote du plan d'eau"}
    line_labels = [rename_dict.get(label, label) for label in line_labels]
    
    ax.legend(
        handles=line_handles,
        labels=line_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),  # décalage sous l’axe X
        frameon=True,
        ncol=1     
    )

    # --- Zones colorées ---
    #zone_labels = ["Cote minimale de restitution",
    #               "Cote permettant de subvenir au besoin d'une saison d'irrigation", 
    #               "Cote permettant de subvenir au besoin de deux saisons d'irrigation",
    #               "Cote normale d'exploitation"]
    #zone_patches = [
    #    mpatches.Patch(color=colors[i], alpha=alpha, label=zone_labels[i])
    #    for i in reversed(range(len(colors)))
    #]
    #leg_zones = ax.legend(handles=zone_patches,
    #                      loc='upper center', bbox_to_anchor=(0.5, -0.20),
    #                      ncol=1, frameon=True)  # une colonne

    # Ajouter les deux légendes à l'axe
    #ax.add_artist(leg_zones)
    
    ax.set_title(f"Barrage des Olivettes \nProjection des cotes jusqu'à fin {last_date.year}")
    ax.grid(True)

    # --- Scatter invisible pour interactivité ---
    if zoom:
        scatter_obs = ax.scatter(
            data_to_plot.index,
            data_to_plot[value_col],
            s=50,
            facecolor="none",
            edgecolor="none",
        picker=True
        )
    else:
        scatter_obs = ax.scatter(
            data.index,
            data[value_col],
            s=50,
            facecolor="none",
            edgecolor="none",
            picker=True
        )

    scatter_pred = ax.scatter(
        forecast_df.index,
        forecast_df[value_col],
        s=50,
        facecolor="none",
        edgecolor="none",
        picker=True
    )

    # --- Toolbar Matplotlib ---
    for widget in frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    toolbar_frame = tb.Frame(frame)
    toolbar_frame.pack(fill="x")
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()

    # --- Cursor interactif avec tooltip ---
    global active_cursor
    active_cursor = mplcursors.cursor([scatter_obs, scatter_pred], hover=True)

    @active_cursor.connect("add")
    def on_hover(sel):
        x, y = sel.target
        date_dt = mdates.num2date(x)
        date_str = date_dt.strftime("%d/%m/%Y")
        pourcentagetotal = cote_to_vol(y)/ cote_to_vol(163) * 100
        pourcentage = (cote_to_vol(y) - cote_to_vol(152)) / (cote_to_vol(163) - cote_to_vol(152)) * 100
        sel.annotation.set_text(
            f"{date_str}\n{y:.2f} mNGF (soit {cote_to_vol(y)/1e6:.3f} Mm³)\n"
            f"{(cote_to_vol(y) - cote_to_vol(152))/1e6:.3f} Mm³ exploitables\n"
            f"Taux de remplissage {pourcentagetotal:.2f}%\n{pourcentage:.2f}% du volume exploitable"
        )
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

    set_current_figure(fig, active_cursor)

    


# === Interface Tkinter avec Flatly ===
root = tb.Window(themename="flatly")
root.title("Olivettes - Analyse et Prévision")
root.geometry("1200x700")

frame_left_container = tb.Frame(root)
frame_left_container.pack(side="left", fill="y")

# Canvas dans ce frame
canvas_left = tk.Canvas(frame_left_container, borderwidth=0)
canvas_left.pack(side="left", fill="y", expand=True)

# Scrollbar à droite du canvas
scrollbar_left = tb.Scrollbar(frame_left_container, orient="vertical", command=canvas_left.yview)
scrollbar_left.pack(side="left", fill="y")  # ou side="right" si tu veux que ce soit vraiment à droite du Canvas
canvas_left.configure(yscrollcommand=scrollbar_left.set)

# Frame interne qui contiendra tout le contenu de frame_left
frame_left_internal = tb.Frame(canvas_left, padding=10)
canvas_left.create_window((0,0), window=frame_left_internal, anchor="nw")

# Fonction pour mettre à jour scrollregion
def on_frame_configure(event):
    canvas_left.configure(scrollregion=canvas_left.bbox("all"))

frame_left_internal.bind("<Configure>", on_frame_configure)
frame_right = tb.Frame(root)
frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


# --- Section Données ---
frame_data = tb.Labelframe(frame_left_internal, text="Données", padding=10)
frame_data.pack(fill=tk.X, pady=10)


btn_load = tb.Button(frame_data, text="Charger CSV", bootstyle=PRIMARY, command=load_csv)
btn_load.pack(fill=tk.X, pady=5)


lbl_start = tb.Label(frame_data, text="Date de début :")
lbl_start.pack(anchor=W)
entry_start_date = tb.Entry(frame_data)
entry_start_date.insert(0, "01/01/2007")
entry_start_date.pack(fill=tk.X, pady=5)


btn_plot = tb.Button(
    frame_data,
    text="Préparer les données",
    bootstyle=INFO,
    command=lambda: plot_cote(frame_right, start_date=entry_start_date.get())
)
btn_plot.pack(fill=tk.X, pady=5)


# --- Section Seuils ---
frame_thresholds = tb.Labelframe(frame_left_internal, text="Seuils", padding=10)
frame_thresholds.pack(fill=tk.X, pady=10)

seuils_info = [
    (152, "Cote minimale de restitution", "#FF9191","#C55353"),
    (157, "Cote permettant de subvenir au besoin\nd'une saison d'irrigation", "#FFD991","#C57F00"),
    (160.25, "Cote permettant de subvenir au besoin\nde deux saisons d'irrigation", "#FFFF91","#C5C500"),
    (163, "Cote normale d'exploitation", "#B2DBA9","#427A42")
]

thresholds_vars = []
for i, (val, _, _, _) in enumerate(seuils_info, start=1):
    lbl = tb.Label(frame_thresholds, text=f"S{i}")
    lbl.grid(row=0, column=i*2-2, padx=2, pady=2, sticky="e")
    var = tk.DoubleVar(value=val)
    tb.Entry(frame_thresholds, textvariable=var, width=6).grid(row=0, column=i*2-1, padx=2, pady=2)
    thresholds_vars.append(var)

# Frame dépliable pour les infos supplémentaires
frame_info = tk.Frame(frame_thresholds)
for i, (val, info_text,color,bordercol) in enumerate(seuils_info):
    # Label du seuil
    lbl = tk.Label(frame_info, text=f"S{i+1}", width=3)
    lbl.grid(row=i, column=0, padx=2, pady=2, sticky="w")
    
    # Carré couleur
    color_canvas = tk.Canvas(frame_info, width=15, height=15, highlightthickness=0)
    color_canvas.grid(row=i, column=1, padx=2, pady=2)
    color_canvas.create_rectangle(0, 0, 15, 15, fill=color, outline=bordercol)

    # Texte explicatif
    info_label = tk.Label(frame_info, text=info_text, wraplength=400, justify="left", foreground="gray")
    info_label.grid(row=i, column=2, padx=2, pady=2, sticky="w")

# Fonction de dépliage
def toggle_info():
    if frame_info.winfo_viewable():
        frame_info.grid_remove()
        toggle_btn.config(text="+")
    else:
        frame_info.grid(row=1, column=0, columnspan=9, sticky="w", pady=2)
        toggle_btn.config(text="-")

# Bouton + à droite du dernier Entry
toggle_btn = tb.Button(frame_thresholds, text="+", width=2, command=toggle_info)
toggle_btn.grid(row=0, column=len(seuils_info)*2, padx=5)

# --- Paramètres HSV et dicts par défaut ---

def resource_path(relative_path):
    """Retourne le chemin absolu, compatible avec PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        # Quand l’app est packagée par PyInstaller
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        # Quand on lance le script en mode normal
        return os.path.join(os.path.abspath("."), relative_path)

cote_to_vol, vol_to_cote, cote_to_surf = make_hsv_converters(resource_path("HSV_32.txt"))


# --- Section Hypothèses de simulation ---
frame_params = tb.Labelframe(frame_left_internal, text="Hypothèses de simulation", padding=10)
frame_params.pack(fill=tk.X, pady=10)

months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", 
          "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

debit_vars = {}
evap_vars = {}

# En-têtes colonnes
tb.Label(frame_params, text="Mois", width=6, anchor="center").grid(row=0, column=0, padx=2, pady=2)
tb.Label(frame_params, text="Débit (m³/s)", anchor="center").grid(row=0, column=1, padx=30, pady=2)
tb.Label(frame_params, text="Évaporation (mm/j)", anchor="center").grid(row=0, column=2, columnspan=2, padx=20, pady=2)

# Options prédéfinies pour l'évaporation
evap_choices = ["0", "2", "5", "9", "Manuel"]

for i, mois in enumerate(months, start=1):
    # Label du mois
    tb.Label(frame_params, text=mois, width=6).grid(row=i, column=0, padx=2, pady=2)

    # Débit sortant
    d_var = tk.DoubleVar(value=0.0)
    tb.Entry(frame_params, textvariable=d_var, width=8).grid(row=i, column=1, padx=2, pady=2)
    debit_vars[i] = d_var

    # Évaporation
    e_var = tk.DoubleVar(value=0.0)
    evap_vars[i] = e_var  # stocke la valeur finale (menu ou manuel)

    # Menu déroulant
    evap_choice = tk.StringVar(value="0")

    def make_evap_menu(row=i, var=e_var, choice_var=evap_choice):
        # Menu
        menu = tb.Combobox(frame_params, textvariable=choice_var, values=evap_choices, width=6, state="readonly")
        menu.grid(row=row, column=2, padx=2, pady=2)

        # Champ manuel (caché au départ)
        manual_entry = tb.Entry(frame_params, textvariable=var, width=6, state="disabled")
        manual_entry.grid(row=row, column=3, padx=2, pady=2)

        def on_select(event=None):
            if choice_var.get() == "Manuel":
                manual_entry.config(state="normal")
                var.set(0.0)  # valeur par défaut
            else:
                manual_entry.config(state="disabled")
                var.set(float(choice_var.get()))

        menu.bind("<<ComboboxSelected>>", on_select)

    make_evap_menu()


# --- Section Prévisions ---
frame_forecast = tb.Labelframe(frame_left_internal, text="Prévisions", padding=10)
frame_forecast.pack(fill=tk.X, pady=10)



def get_params_dicts():
    debit_sortant_dict = {m: debit_vars[m].get() for m in range(1, 13)}
    evap_dict = {m: evap_vars[m].get() for m in range(1, 13)}
    return debit_sortant_dict, evap_dict


def get_adjustments_from_entries():
    adj = {}
    date_str = entry_date.get().strip()
    value_str = entry_value.get().strip()
    if date_str and value_str:
        try:
            date = pd.to_datetime(date_str, format="%d/%m/%Y")
            value = float(value_str)
            adj[date] = value
        except Exception:
            messagebox.showwarning("Ajustement ignoré", "Format de date ou valeur invalide. Ajustement non pris en compte.")
    return adj

btn_forecast = tb.Button(frame_forecast, text="Prévoir", bootstyle=SUCCESS,
                            command=lambda: forecast_volume(
                                data, cote_to_vol, vol_to_cote, cote_to_surf,
                                *get_params_dicts(),
                                frame_right,
                                thresholds=[v.get() for v in thresholds_vars],
                                colors=['red', 'orange', 'yellow', 'green'],
                                alpha=0.2,
                                zoom=False,
                             adjustments=get_adjustments_from_entries()))
btn_forecast.pack(fill=tk.X, pady=5)


btn_zoom = tb.Button(frame_forecast, text="Zoom Prévision", bootstyle=WARNING,
                        command=lambda: forecast_volume(
                            data, cote_to_vol, vol_to_cote, cote_to_surf,
                            *get_params_dicts(),
                            frame_right,
                            thresholds=[v.get() for v in thresholds_vars],
                            colors=['red', 'orange', 'yellow', 'green'],
                            alpha=0.2,
                            zoom=True,
                             adjustments=get_adjustments_from_entries()))
btn_zoom.pack(fill=tk.X, pady=5)

adjustments = {}

# --- Frame pour ajustement ---
frame_adjust = tb.Labelframe(frame_forecast, text="Ajuster", padding=10)
frame_adjust.pack(fill=tk.X, pady=5)


# --- Date
lbl_date = tb.Label(frame_adjust, text="Date (JJ/MM/AAAA) :")
lbl_date.grid(row=0, column=0, padx=5, pady=2)

entry_date = tb.Entry(frame_adjust)
entry_date.grid(row=0, column=1, padx=5, pady=2)

# Valeur
lbl_value = tb.Label(frame_adjust, text="Ajustement de cote (en m) :")
lbl_value.grid(row=1, column=0, padx=5, pady=2)
entry_value = tb.Entry(frame_adjust, width=10)
entry_value.grid(row=1, column=1, padx=5, pady=2)


active_cursor = None  # variable globale pour mplcursors
fig_global = None     # variable globale pour le figure actuelle

def set_current_figure(fig, cursor):
    global fig_global, active_cursor
    fig_global = fig
    active_cursor = cursor

def save_figure():
    global fig_global
    if fig_global is None:
        messagebox.showwarning("Attention", "Aucun graphique à enregistrer.")
        return
    filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG","*.png"), ("PDF","*.pdf"), ("JPEG","*.jpg")])
    if filepath:
        fig_global.savefig(filepath, bbox_inches='tight')
        messagebox.showinfo("Succès", f"Graphique enregistré sous : {filepath}")

def remove_cursor():
    global active_cursor
    if active_cursor is not None:
        active_cursor.remove()
        active_cursor = None

# Frame contenant les boutons
btn_frame_forecast = tb.Frame(frame_forecast)
btn_frame_forecast.pack(fill="x", pady=5)


# Bouton pour sauvegarder le graphique
btn_save = tb.Button(btn_frame_forecast, text="Enregistrer", bootstyle="info", command=save_figure)
btn_save.pack(side="left", fill="x", expand=True, padx=2, pady=2)

# Bouton pour retirer la bulle
btn_remove_bubble = tb.Button(btn_frame_forecast, text="Retirer la bulle", bootstyle="danger", command=remove_cursor)
btn_remove_bubble.pack(side="left", fill="x", expand=True, padx=2, pady=2)


## Bouton Ajouter
#def add_adjustment(zoom=False):
#    # récupérer et convertir la date
#    try:
#        date = pd.to_datetime(entry_date.get(), format="%d/%m/%Y")
#    except Exception:
#        messagebox.showerror("Erreur", "Format de date invalide (JJ/MM/AAAA)")
#        return
#    
#    # récupérer et convertir la valeur
#    try:
#        value = float(entry_value.get())
#    except ValueError:
#        messagebox.showerror("Erreur", "Valeur invalide pour l'ajustement")
#        return
#    
#    # mettre à jour le dictionnaire
#    adjustments[date] = value
#    
#    # relancer le forecast automatiquement
#    forecast_volume(
#        data, cote_to_vol, vol_to_cote, cote_to_surf,
#        *get_params_dicts(),
#        frame_right,
#        thresholds=[v.get() for v in thresholds_vars],
#        colors=['red', 'orange', 'yellow', 'green'],
#        alpha=0.2,
#        zoom=zoom,
#        adjustments=adjustments
#    )
#
#btn_add_adj = tb.Button(frame_adjust, text="Ajuster", bootstyle=INFO, command=lambda: add_adjustment(zoom=True))
#btn_add_adj.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")


root.protocol("WM_DELETE_WINDOW", on_closing)  
root.mainloop()


# ligne de commande export app:
# pyinstaller --onefile --noconsole app_olivettes.py --add-data "HSV_32.txt;." --add-data "C:\Users\XXXXXX\AppData\Local\Programs\Python\Python313\Lib\site-packages\matplotlib\mpl-data;matplotlib\mpl-data" --hidden-import=matplotlib.backends.backend_pdf