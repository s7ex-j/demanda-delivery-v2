"""
Utilidad para exportar automáticamente los outputs relevantes de los
notebooks a archivos de texto en reports/notebook_outputs/, para que
los resultados queden documentados sin depender de exportación manual.
"""

import os


def guardar_output(nombre_archivo: str, contenido, carpeta: str = '../reports/notebook_outputs') -> None:
    """Guarda `contenido` (se convierte a str) en
    `carpeta/nombre_archivo.txt`. Crea la carpeta si no existe.

    Uso típico dentro de un notebook:
        guardar_output('03b_modelo_panel', resultado_panel)
    """
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f'{nombre_archivo}.txt')
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(str(contenido))
    print(f'Output guardado en: {ruta}')


# ======================================================================
# Captura COMPLETA de todos los outputs del notebook (para exportar la
# corrida entera a un solo .txt combinado).
# ======================================================================

import io
import sys as _sys


class _Tee(io.TextIOBase):
    """Escribe simultáneamente en varios streams (p. ej. la salida
    normal del notebook Y un buffer en memoria)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


_buffer_captura = None
_stdout_original = None


def iniciar_captura_completa() -> None:
    """Llamar en la PRIMERA celda de código del notebook.

    Redirige stdout para que todo lo impreso con print() se escriba
    también a un buffer en memoria, y registra un hook de IPython
    ('post_run_cell') para que el resultado de celda que Jupyter
    muestra automáticamente (p. ej. escribir `resultado_iv.first_stage`
    sola en una celda, sin print()) también quede capturado -- no solo
    lo que se imprime explícitamente.
    """
    global _buffer_captura, _stdout_original

    _buffer_captura = io.StringIO()
    _stdout_original = _sys.stdout
    _sys.stdout = _Tee(_stdout_original, _buffer_captura)

    try:
        from IPython import get_ipython
        ip = get_ipython()

        def _capturar_resultado_celda(result):
            if result.result is not None:
                print(result.result)

        ip.events.register('post_run_cell', _capturar_resultado_celda)
    except Exception:
        # Si no hay IPython disponible (p. ej. corriendo como script
        # plano), la captura de stdout sigue funcionando igual; solo
        # se pierde la captura de resultados de celda "desnudos".
        pass

    print('--- Captura de outputs iniciada ---')


def guardar_captura_completa(nombre_archivo: str, carpeta: str = '../reports/notebook_outputs') -> None:
    """Llamar en la ÚLTIMA celda del notebook. Guarda todo lo capturado
    desde iniciar_captura_completa() en un único archivo de texto.
    """
    if _buffer_captura is None:
        print('Advertencia: no se llamó a iniciar_captura_completa() al inicio del notebook.')
        return

    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f'{nombre_archivo}.txt')
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(_buffer_captura.getvalue())

    # Este print va también al buffer (ya guardado), así que se ve en
    # el notebook pero no hace falta que quede en el archivo.
    _sys.stdout = _stdout_original
    print(f'Captura completa guardada en: {ruta}')
