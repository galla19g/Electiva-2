import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

# Parámetros iniciales
amplitud_inicial = 1.0
frecuencia_inicial = 1.0
fase_inicial = 0.0
nivel_ruido_inicial = 0.2
ruido_activado = False

# Vector de tiempo (de 0 a 10 segundos, 1000 puntos)
t = np.linspace(0, 10, 1000)

def generar_onda(A, f, fase):
    """Genera la onda senoidal base"""
    return A * np.sin(2 * np.pi * f * t + fase)

def agregar_ruido(senal, desviacion_estandar):
    """Agrega ruido gaussiano a la señal"""
    ruido = np.random.normal(0, desviacion_estandar, senal.shape)
    return senal + ruido

# Crear la figura y los ejes principales para la gráfica
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(left=0.1, bottom=0.35) # Dejar espacio en la parte inferior para los controles

# Generar señales iniciales
senal_original = generar_onda(amplitud_inicial, frecuencia_inicial, fase_inicial)
senal_con_ruido = agregar_ruido(senal_original, nivel_ruido_inicial)

# Dibujar las líneas en la gráfica
linea_original, = ax.plot(t, senal_original, label='Señal Original', color='blue', lw=2)
linea_ruido, = ax.plot(t, senal_con_ruido, label='Señal con Ruido', color='red', alpha=0.6)

# Ocultar la señal con ruido inicialmente
linea_ruido.set_visible(ruido_activado)

# Configuración visual de la gráfica
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Amplitud')
ax.set_title('Análisis de Onda Electromagnética y Ruido')
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_ylim(-4, 4)

# --- CREACIÓN DE CONTROLES INTERACTIVOS (WIDGETS) ---

color_ejes = 'lightgoldenrodyellow'

# Definir la posición de los sliders [izq, abajo, ancho, alto]
eje_amp = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=color_ejes)
eje_freq = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=color_ejes)
eje_fase = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=color_ejes)
eje_ruido = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=color_ejes)

# Crear los Sliders (Deslizadores)
slider_amp = Slider(eje_amp, 'Amplitud', 0.1, 5.0, valinit=amplitud_inicial)
slider_freq = Slider(eje_freq, 'Frecuencia', 0.1, 5.0, valinit=frecuencia_inicial)
slider_fase = Slider(eje_fase, 'Fase (rad)', 0.0, 2 * np.pi, valinit=fase_inicial)
slider_ruido = Slider(eje_ruido, 'Nivel Ruido', 0.0, 2.0, valinit=nivel_ruido_inicial)

# Crear Checkbox (Casilla de verificación) para activar/desactivar ruido
eje_check = plt.axes([0.85, 0.1, 0.1, 0.15])
check_ruido = CheckButtons(eje_check, ['Activar\nRuido'], [ruido_activado])

# Función para actualizar la gráfica cuando se mueve un slider
def actualizar_sliders(val):
    amp = slider_amp.val
    freq = slider_freq.val
    fase = slider_fase.val
    nivel_ruido = slider_ruido.val
    
    # Recalcular la onda original
    nueva_original = generar_onda(amp, freq, fase)
    linea_original.set_ydata(nueva_original)
    
    # Recalcular la onda con ruido
    # Se genera nuevo ruido en cada actualización para dar la sensación de "tiempo real / estática"
    nueva_con_ruido = agregar_ruido(nueva_original, nivel_ruido)
    linea_ruido.set_ydata(nueva_con_ruido)
    
    # Ajustar dinámicamente los límites del eje Y si la amplitud crece mucho
    max_val = max(amp + nivel_ruido, 4)
    ax.set_ylim(-max_val, max_val)
    
    fig.canvas.draw_idle()

# Función para actualizar la visibilidad cuando se clickea el checkbox
def actualizar_checkbox(label):
    estado_actual = linea_ruido.get_visible()
    linea_ruido.set_visible(not estado_actual)
    fig.canvas.draw_idle()

# Conectar los eventos de los widgets con sus funciones de actualización
slider_amp.on_changed(actualizar_sliders)
slider_freq.on_changed(actualizar_sliders)
slider_fase.on_changed(actualizar_sliders)
slider_ruido.on_changed(actualizar_sliders)
check_ruido.on_clicked(actualizar_checkbox)

# Mostrar la interfaz interactiva
plt.show()
