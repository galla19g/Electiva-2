"""
=============================================================
  Visualizador Interactivo de Ondas con Ruido
  Materia: Electiva 2
=============================================================
Descripción:
  Aplicación de escritorio (Matplotlib + Tkinter) que permite
  explorar de forma interactiva una onda periódica y el efecto
  del ruido sobre ella.  Todos los parámetros se actualizan en
  tiempo real mediante sliders y radio-botones.

Parámetros configurables:
  · Amplitud                (slider)
  · Frecuencia [Hz]         (slider, enlazado con Período)
  · Período [s]             (slider, enlazado con Frecuencia)
  · Frecuencia de muestreo  (slider – puntos por segundo)
  · Nivel de ruido (σ)      (slider)
  · Activar / desactivar ruido (checkbox)
  · Tipo de onda: Senoidal / Cuadrada / Triangular (radio)
  · Tipo de ruido: Gaussiano / Uniforme (radio)

Extras implementados:
  · Botón "Resetear" → restaura valores por defecto
  · Botón "Guardar"  → guarda la figura como PNG
  · SNR calculado y mostrado en dB
  · Información numérica de parámetros en la gráfica
  · Leyenda, cuadrícula, tema oscuro
=============================================================
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONSTANTES Y VALORES POR DEFECTO
# ─────────────────────────────────────────────────────────────
DURACION = 2.0          # segundos visibles de forma fija en el eje X

DEFAULTS = {
    "amplitud":    1.0,
    "frecuencia":  2.0,   # Hz
    "fs":          500,   # puntos por segundo (frecuencia de muestreo)
    "ruido_std":   0.3,   # desviación estándar del ruido
    "ruido_activo": False,
    "tipo_onda":   "Senoidal",
    "tipo_ruido":  "Gaussiano",
}

TIPOS_ONDA  = ["Senoidal", "Cuadrada", "Triangular"]
TIPOS_RUIDO = ["Gaussiano", "Uniforme"]

# Paleta de colores (tema oscuro)
CLR_BG       = "#1e1e2e"   # fondo figura
CLR_PLOT_BG  = "#2a2a3e"   # fondo gráfico
CLR_ORIG     = "#4fc3f7"   # azul claro → onda original
CLR_RUIDO    = "#ef9a9a"   # rosa      → onda con ruido
CLR_SLIDER   = "#4fc3f7"
CLR_SRUIDO   = "#ef9a9a"
CLR_TEXT     = "white"
CLR_BTN      = "#2d2d50"
CLR_BTN_HOV  = "#4444aa"


# ─────────────────────────────────────────────────────────────
#  FUNCIONES DE SEÑAL
# ─────────────────────────────────────────────────────────────

def generar_tiempo(fs: float) -> np.ndarray:
    """
    Genera el vector de tiempo para una duración fija (DURACION seg.)
    con la frecuencia de muestreo indicada.

    Args:
        fs: puntos por segundo (frecuencia de muestreo).
    Returns:
        Array 1-D de tiempos en segundos.
    """
    n_puntos = max(int(fs * DURACION), 100)  # mínimo 100 puntos
    return np.linspace(0, DURACION, n_puntos)


def generar_onda(t: np.ndarray, A: float, f: float, tipo: str) -> np.ndarray:
    """
    Genera la onda de acuerdo al tipo seleccionado.

    Args:
        t:    vector de tiempo.
        A:    amplitud.
        f:    frecuencia en Hz.
        tipo: 'Senoidal', 'Cuadrada' o 'Triangular'.
    Returns:
        Array con los valores de la señal.
    """
    fase = 2 * np.pi * f * t
    if tipo == "Cuadrada":
        return A * np.sign(np.sin(fase))
    elif tipo == "Triangular":
        # Aproximación analítica exacta con arcsin
        return A * (2 / np.pi) * np.arcsin(np.sin(fase))
    else:  # Senoidal (default)
        return A * np.sin(fase)


def agregar_ruido(senal: np.ndarray, std: float, tipo: str) -> np.ndarray:
    """
    Contamina la señal con ruido del tipo indicado.

    Args:
        senal: señal original.
        std:   desviación estándar del ruido (intensidad).
        tipo:  'Gaussiano' o 'Uniforme'.
    Returns:
        Señal contaminada.
    """
    n = senal.shape[0]
    if tipo == "Uniforme":
        # Para ruido uniforme con la misma std: rango = std * sqrt(12)
        limite = std * np.sqrt(3)
        ruido = np.random.uniform(-limite, limite, n)
    else:  # Gaussiano
        ruido = np.random.normal(0, std, n)
    return senal + ruido


def calcular_snr_db(senal_pura: np.ndarray, senal_ruidosa: np.ndarray) -> float:
    """
    Calcula la Relación Señal / Ruido (SNR) en decibelios.

    SNR_dB = 10 * log10(P_señal / P_ruido)

    Args:
        senal_pura:    señal sin ruido.
        senal_ruidosa: señal con ruido superpuesto.
    Returns:
        SNR en dB (float). Retorna inf si no hay ruido.
    """
    potencia_senal = np.mean(senal_pura ** 2)
    ruido          = senal_ruidosa - senal_pura
    potencia_ruido = np.mean(ruido ** 2)
    if potencia_ruido < 1e-12:
        return float("inf")
    return 10 * np.log10(potencia_senal / potencia_ruido)


# ─────────────────────────────────────────────────────────────
#  CONSTRUCCIÓN DE LA FIGURA
# ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(13, 9))
fig.patch.set_facecolor(CLR_BG)
fig.canvas.manager.set_window_title("Visualizador Interactivo de Ondas con Ruido")

# Área de la gráfica principal (parte superior)
plt.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.52)
ax = fig.add_subplot(111)
ax.set_facecolor(CLR_PLOT_BG)
ax.set_xlabel("Tiempo [s]", color=CLR_TEXT, fontsize=10)
ax.set_ylabel("Amplitud",   color=CLR_TEXT, fontsize=10)
ax.set_title(
    "Visualizador Interactivo de Ondas con Ruido",
    color=CLR_TEXT, fontsize=13, fontweight="bold", pad=10
)
ax.tick_params(colors=CLR_TEXT)
for spine in ax.spines.values():
    spine.set_color("#555577")
ax.grid(True, linestyle="--", alpha=0.3, color="#aaaaaa")

# ── Señales iniciales ────────────────────────────────────────
t_vec        = generar_tiempo(DEFAULTS["fs"])
sig_orig     = generar_onda(t_vec, DEFAULTS["amplitud"], DEFAULTS["frecuencia"], DEFAULTS["tipo_onda"])
sig_ruidosa  = agregar_ruido(sig_orig, DEFAULTS["ruido_std"], DEFAULTS["tipo_ruido"])

linea_orig,  = ax.plot(t_vec, sig_orig,    color=CLR_ORIG,  lw=2,   label="Original",  zorder=3)
linea_ruido, = ax.plot(t_vec, sig_ruidosa, color=CLR_RUIDO, lw=1,   label="Con Ruido", zorder=2, alpha=0.80)
linea_ruido.set_visible(DEFAULTS["ruido_activo"])

ax.legend(loc="upper right", facecolor=CLR_PLOT_BG, edgecolor="#555577",
          labelcolor=CLR_TEXT, fontsize=9)

# Texto de parámetros en el interior del gráfico
txt_params = ax.text(
    0.01, 0.97, "",
    transform=ax.transAxes, va="top", ha="left",
    fontsize=8, color="#ccccff",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#11112a", alpha=0.75, edgecolor="#555577")
)


# ─────────────────────────────────────────────────────────────
#  FUNCIÓN AUXILIAR PARA CREAR EJES DE CONTROL
# ─────────────────────────────────────────────────────────────

def nuevo_eje(left, bottom, width=0.32, height=0.025) -> plt.Axes:
    """Crea y devuelve un Axes para un widget, con fondo oscuro."""
    eje = fig.add_axes([left, bottom, width, height])
    eje.set_facecolor(CLR_PLOT_BG)
    return eje


# ─────────────────────────────────────────────────────────────
#  SLIDERS
# ─────────────────────────────────────────────────────────────

# Columna izquierda
sld_amp  = Slider(nuevo_eje(0.07, 0.45), "Amplitud",         0.1,  5.0,  valinit=DEFAULTS["amplitud"],              color=CLR_SLIDER)
sld_freq = Slider(nuevo_eje(0.07, 0.39), "Frecuencia (Hz)",  0.1, 10.0,  valinit=DEFAULTS["frecuencia"],            color=CLR_SLIDER)
sld_per  = Slider(nuevo_eje(0.07, 0.33), "Período (s)",      0.1, 10.0,  valinit=1.0 / DEFAULTS["frecuencia"],     color=CLR_SLIDER)
sld_fs   = Slider(nuevo_eje(0.07, 0.27), "Muestreo (pts/s)",  50, 2000,  valinit=DEFAULTS["fs"], valfmt="%d",      color=CLR_SLIDER)

# Columna derecha
sld_ruido = Slider(nuevo_eje(0.55, 0.45), "Nivel de Ruido (σ)", 0.0, 3.0, valinit=DEFAULTS["ruido_std"], color=CLR_SRUIDO)

# Estilo de etiquetas y valores
for s in [sld_amp, sld_freq, sld_per, sld_fs, sld_ruido]:
    s.label.set_color(CLR_TEXT)
    s.valtext.set_color(CLR_ORIG)


# ─────────────────────────────────────────────────────────────
#  RADIO BUTTONS – Tipo de onda
# ─────────────────────────────────────────────────────────────

ax_radio_onda = fig.add_axes([0.55, 0.26, 0.17, 0.16])
ax_radio_onda.set_facecolor(CLR_BG)
ax_radio_onda.set_title("Tipo de Onda", color=CLR_TEXT, fontsize=8, pad=4)
radio_onda = RadioButtons(ax_radio_onda, TIPOS_ONDA, activecolor=CLR_ORIG)
for lbl in radio_onda.labels:
    lbl.set_color(CLR_TEXT)
    lbl.set_fontsize(8)


# ─────────────────────────────────────────────────────────────
#  RADIO BUTTONS – Tipo de ruido
# ─────────────────────────────────────────────────────────────

ax_radio_ruido = fig.add_axes([0.74, 0.26, 0.17, 0.13])
ax_radio_ruido.set_facecolor(CLR_BG)
ax_radio_ruido.set_title("Tipo de Ruido", color=CLR_TEXT, fontsize=8, pad=4)
radio_ruido = RadioButtons(ax_radio_ruido, TIPOS_RUIDO, activecolor=CLR_SRUIDO)
for lbl in radio_ruido.labels:
    lbl.set_color(CLR_TEXT)
    lbl.set_fontsize(8)


# ─────────────────────────────────────────────────────────────
#  CHECKBOX – Activar / desactivar ruido
# ─────────────────────────────────────────────────────────────

ax_check = fig.add_axes([0.55, 0.19, 0.17, 0.06])
ax_check.set_facecolor(CLR_BG)
check_ruido = CheckButtons(ax_check, ["Activar Ruido"], [DEFAULTS["ruido_activo"]])
check_ruido.labels[0].set_color(CLR_TEXT)
check_ruido.labels[0].set_fontsize(9)


# ─────────────────────────────────────────────────────────────
#  BOTONES – Resetear y Guardar
# ─────────────────────────────────────────────────────────────

ax_btn_reset = fig.add_axes([0.74, 0.19, 0.08, 0.04])
ax_btn_save  = fig.add_axes([0.84, 0.19, 0.08, 0.04])

btn_reset = Button(ax_btn_reset, "↺ Resetear", color=CLR_BTN, hovercolor=CLR_BTN_HOV)
btn_save  = Button(ax_btn_save,  "💾 Guardar",  color=CLR_BTN, hovercolor=CLR_BTN_HOV)

btn_reset.label.set_color(CLR_TEXT)
btn_save.label.set_color(CLR_TEXT)


# ─────────────────────────────────────────────────────────────
#  ÁREA SNR
# ─────────────────────────────────────────────────────────────

ax_snr = fig.add_axes([0.55, 0.13, 0.40, 0.05])
ax_snr.set_facecolor(CLR_BG)
ax_snr.axis("off")
txt_snr = ax_snr.text(
    0.5, 0.5, "SNR: — dB",
    transform=ax_snr.transAxes, ha="center", va="center",
    fontsize=11, fontweight="bold", color="#ffdd88"
)


# ─────────────────────────────────────────────────────────────
#  ESTADO INTERNO (tipo de onda / ruido)
# ─────────────────────────────────────────────────────────────

# Usamos un dict mutable para guardar los estados que no son sliders
estado = {
    "tipo_onda":  DEFAULTS["tipo_onda"],
    "tipo_ruido": DEFAULTS["tipo_ruido"],
}

# Bandera para evitar recursión al sincronizar freq ↔ período
_sincronizando = [False]


# ─────────────────────────────────────────────────────────────
#  FUNCIÓN CENTRAL DE ACTUALIZACIÓN
# ─────────────────────────────────────────────────────────────

def actualizar(_=None):
    """
    Lee todos los controles, recalcula las señales y actualiza
    la gráfica.  Se llama automáticamente al interactuar con
    cualquier widget.
    """
    A     = sld_amp.val
    f     = sld_freq.val
    fs    = int(sld_fs.val)
    std   = sld_ruido.val
    tipo_o = estado["tipo_onda"]
    tipo_r = estado["tipo_ruido"]
    ruido_visible = linea_ruido.get_visible()

    # ── Regenerar señales ────────────────────────────────────
    t_nuevo      = generar_tiempo(fs)
    nueva_orig   = generar_onda(t_nuevo, A, f, tipo_o)
    nueva_ruido  = agregar_ruido(nueva_orig, std, tipo_r)

    linea_orig.set_xdata(t_nuevo)
    linea_orig.set_ydata(nueva_orig)
    linea_ruido.set_xdata(t_nuevo)
    linea_ruido.set_ydata(nueva_ruido)

    # ── Eje X → siempre 0 … DURACION (la compresión/expansión
    #    horizontal se aprecia al ver más o menos ciclos) ─────
    ax.set_xlim(0, DURACION)

    # ── Eje Y → dinámico según amplitud + margen de ruido ────
    if ruido_visible:
        margen = A + std * 1.5 + 0.2
    else:
        margen = A * 1.2 + 0.2
    ax.set_ylim(-max(margen, 0.5), max(margen, 0.5))

    # ── SNR ─────────────────────────────────────────────────
    if ruido_visible and std > 1e-6:
        snr = calcular_snr_db(nueva_orig, nueva_ruido)
        txt_snr.set_text(f"SNR efectivo: {snr:.2f} dB")
    else:
        txt_snr.set_text("SNR: ∞ dB  (sin ruido activo)")

    # ── Texto de parámetros en el gráfico ───────────────────
    periodo = 1.0 / f if f > 1e-9 else float("inf")
    info = (
        f"A = {A:.2f}    f = {f:.2f} Hz    T = {periodo:.4f} s\n"
        f"Muestreo = {fs} pts/s    σ_ruido = {std:.2f}    "
        f"Onda: {tipo_o}    Ruido: {tipo_r}"
    )
    txt_params.set_text(info)

    fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────
#  CALLBACKS ESPECÍFICOS
# ─────────────────────────────────────────────────────────────

def on_frecuencia_cambia(val):
    """Sincroniza el slider de período cuando cambia la frecuencia."""
    if _sincronizando[0]:
        return
    _sincronizando[0] = True
    sld_per.set_val(round(1.0 / max(val, 1e-6), 5))
    _sincronizando[0] = False
    actualizar()


def on_periodo_cambia(val):
    """Sincroniza el slider de frecuencia cuando cambia el período."""
    if _sincronizando[0]:
        return
    _sincronizando[0] = True
    sld_freq.set_val(round(1.0 / max(val, 1e-6), 5))
    _sincronizando[0] = False
    actualizar()


def on_checkbox(_label):
    """Activa o desactiva la visibilidad de la onda con ruido."""
    linea_ruido.set_visible(not linea_ruido.get_visible())
    actualizar()


def on_tipo_onda(label):
    """Cambia el tipo de onda seleccionado."""
    estado["tipo_onda"] = label
    actualizar()


def on_tipo_ruido(label):
    """Cambia el tipo de ruido seleccionado."""
    estado["tipo_ruido"] = label
    actualizar()


def on_resetear(_event):
    """Restaura todos los parámetros a sus valores por defecto."""
    _sincronizando[0] = True
    sld_amp.set_val(DEFAULTS["amplitud"])
    sld_freq.set_val(DEFAULTS["frecuencia"])
    sld_per.set_val(1.0 / DEFAULTS["frecuencia"])
    sld_fs.set_val(DEFAULTS["fs"])
    sld_ruido.set_val(DEFAULTS["ruido_std"])
    estado["tipo_onda"]  = DEFAULTS["tipo_onda"]
    estado["tipo_ruido"] = DEFAULTS["tipo_ruido"]
    # Forzar radio buttons a posición inicial
    radio_onda.set_active(TIPOS_ONDA.index(DEFAULTS["tipo_onda"]))
    radio_ruido.set_active(TIPOS_RUIDO.index(DEFAULTS["tipo_ruido"]))
    # Desactivar ruido si estaba activo
    if linea_ruido.get_visible() != DEFAULTS["ruido_activo"]:
        linea_ruido.set_visible(DEFAULTS["ruido_activo"])
    _sincronizando[0] = False
    actualizar()


def on_guardar(_event):
    """Guarda la figura completa como imagen PNG."""
    nombre_archivo = "onda_captura.png"
    fig.savefig(nombre_archivo, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[INFO] Imagen guardada como '{nombre_archivo}'")


# ─────────────────────────────────────────────────────────────
#  CONEXIÓN DE EVENTOS
# ─────────────────────────────────────────────────────────────

sld_amp.on_changed(actualizar)
sld_freq.on_changed(on_frecuencia_cambia)
sld_per.on_changed(on_periodo_cambia)
sld_fs.on_changed(actualizar)
sld_ruido.on_changed(actualizar)

check_ruido.on_clicked(on_checkbox)
radio_onda.on_clicked(on_tipo_onda)
radio_ruido.on_clicked(on_tipo_ruido)

btn_reset.on_clicked(on_resetear)
btn_save.on_clicked(on_guardar)

# ─────────────────────────────────────────────────────────────
#  DIBUJADO INICIAL Y LANZAMIENTO
# ─────────────────────────────────────────────────────────────

actualizar()      # primer renderizado con los valores por defecto
plt.show()
