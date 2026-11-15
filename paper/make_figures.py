"""Figuras y tabla del artículo de ICBDAIA 2027 (protocolo 2 tareas x 50 clases).

Lee solo resultados/plan_gpu/*/seed_*/results.csv (los resultados congelados de la tarea T3)
y escribe en paper_congreso/:
    figuras/fig1_buffer20.{pdf,png}   precisión en A, en B y global (ACC) con buffer del 20 %
    figuras/fig2_buffer.{pdf,png}     ACC según el tamaño del buffer (5, 10 y 20 %)
    figuras/fig3_ablacion.{pdf,png}   retención de A y ACC según la fracción f del lote
    tablas/tabla_resultados.tex       Tabla 2 (buffer del 20 %)
    tablas/valores_figuras.csv        media, desviación típica y n de cada punto dibujado

Uso (desde la raíz del repo):  python paper/make_figures.py
Requiere matplotlib y numpy.
"""

from __future__ import annotations

import csv
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "resultados" / "plan_gpu"
SALIDA = RAIZ / "paper"
FIG = SALIDA / "figures"
TAB = SALIDA / "tables"

# Paleta categórica validada (deuteranopía/protanopía/tritanopía, ΔE ≥ 9,2 entre todos los pares).
# El color sigue al método en las tres figuras; "sin replay" es una referencia neutra.
COLOR = {
    "none": "#8a8984",
    "concat": "#2a78d6",
    "balanced": "#eb6834",
    "erace": "#1baf7a",
}
MARCADOR = {"concat": "o", "balanced": "s", "erace": "^"}
NOMBRE = {
    "none": "No replay",
    "concat": "Conventional mix",
    "balanced": "Balanced batches ($f=0.5$)",
    "erace": "Adapted ER-ACE",
}
# Fig. 3: el tamaño del buffer es ordinal, así que usa dos pasos de un mismo azul.
COLOR_BUFFER = {10: "#7fb0ea", 20: "#1d5bab"}
TINTA = "#1f1e1c"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"

BUFFERS = [5, 10, 20]
FILA = {0: "No Replay", 5: "5% Replay", 10: "10% Replay", 20: "20% Replay"}


def leer(carpeta: str) -> dict[int, dict[str, dict[str, float]]]:
    """{semilla: {fila: {columna: valor}}} con solo las semillas completas (seed_NN)."""
    out: dict[int, dict[str, dict[str, float]]] = {}
    for d in sorted((DATOS / carpeta).iterdir()):
        m = re.fullmatch(r"seed_(\d+)", d.name)
        if not m or not (d / "results.csv").exists():
            continue
        with open(d / "results.csv", newline="", encoding="utf-8") as fh:
            filas = {}
            for r in csv.DictReader(fh):
                filas[r["fraction"]] = {k: float(v) for k, v in r.items() if k != "fraction" and v != ""}
        out[int(m.group(1))] = filas
    if not out:
        raise SystemExit(f"Sin resultados en {DATOS / carpeta}")
    return out


def resumen(datos, buffer: int, col: str) -> tuple[float, float, int]:
    v = [s[FILA[buffer]][col] for s in datos.values() if FILA[buffer] in s]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)


def estilo() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.edgecolor": TINTA_2,
        "axes.labelcolor": TINTA,
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": REJILLA,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })


def guardar(fig, nombre: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{nombre}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# Ancho de texto de la plantilla Springer (122 mm).
ANCHO = 4.8


def fig1(D, valores):
    grupos = [("step4_acc_a", "Task A (classes 0–49)"),
              ("step4_acc_b", "Task B (classes 50–99)"),
              ("acc_all_100_classes", "Overall (ACC)")]
    metodos = [("none", D["concat"], 0), ("concat", D["concat"], 20),
               ("balanced", D["balanced"], 20), ("erace", D["erace"], 20)]
    fig, ax = plt.subplots(figsize=(ANCHO, 2.3))
    ancho = 0.19
    for j, (clave, datos, buf) in enumerate(metodos):
        xs, ms, sds = [], [], []
        for i, (col, _) in enumerate(grupos):
            m, sd, n = resumen(datos, buf, col)
            xs.append(i + (j - 1.5) * (ancho + 0.012))
            ms.append(m)
            sds.append(sd)
            valores.append(["fig1", NOMBRE[clave], buf, col, round(m, 2), round(sd, 2), n])
        ax.bar(xs, ms, width=ancho, color=COLOR[clave], label=NOMBRE[clave],
               edgecolor="white", linewidth=0.8, zorder=2)
        ax.errorbar(xs, ms, yerr=sds, fmt="none", ecolor=TINTA, elinewidth=0.7,
                    capsize=1.8, capthick=0.7, zorder=3)
    # La barra de "sin replay" en A mide 0 y no se ve: se rotula.
    ax.annotate("0.0", (0 - 1.5 * (ancho + 0.012), 0), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5, color=TINTA_2)
    ax.set_xticks(range(len(grupos)), [g[1] for g in grupos])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Accuracy over 100 classes (%)")
    ax.set_ylim(0, 65)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2,
              columnspacing=1.0, borderaxespad=0.2)
    guardar(fig, "fig1_buffer20")


def fig2(D, valores):
    fig, ax = plt.subplots(figsize=(ANCHO * 0.62, 2.3))
    x = list(range(len(BUFFERS)))
    for clave in ("concat", "balanced", "erace"):
        r = [resumen(D[clave], b, "acc_all_100_classes") for b in BUFFERS]
        ms = [v[0] for v in r]
        ax.errorbar(x, ms, yerr=[v[1] for v in r], color=COLOR[clave], marker=MARCADOR[clave],
                    markersize=4.5, linewidth=1.4, elinewidth=0.7, capsize=1.8,
                    markeredgecolor="white", markeredgewidth=0.6, label=NOMBRE[clave], zorder=3)
        ax.annotate(f"{ms[-1]:.1f}", (x[-1], ms[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color=TINTA)
        for b, (m, sd, n) in zip(BUFFERS, r):
            valores.append(["fig2", NOMBRE[clave], b, "acc_all_100_classes", round(m, 2), round(sd, 2), n])
    m0, _, _ = resumen(D["concat"], 0, "acc_all_100_classes")
    ax.axhline(m0, color=COLOR["none"], linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    ax.text(x[0] - 0.15, m0 + 0.6, f"No replay ({m0:.1f})", fontsize=7, color=TINTA_2, va="bottom")
    ax.set_xticks(x, [f"{b}%" for b in BUFFERS])
    ax.set_xlim(-0.25, len(BUFFERS) - 0.55)
    ax.set_xlabel("Buffer size (% of task-A images per client)")
    ax.set_ylabel("Overall accuracy, ACC (%)")
    ax.set_ylim(26, 50)
    ax.legend(loc="upper left", handlelength=1.6)
    guardar(fig, "fig2_buffer")


def fig3(D, valores):
    """Retención de A y ACC frente a f. La fracción natural se dibuja en su valor real."""
    abl = {"natural": "natural", 0.25: "f025", 0.5: "balanced", 0.75: "f075"}
    fig, axes = plt.subplots(1, 2, figsize=(ANCHO, 2.2), sharex=True)
    paneles = [("retention_a_percent", "Retention of task A (%)"),
               ("acc_all_100_classes", "Overall accuracy, ACC (%)")]
    for ax, (col, etiqueta) in zip(axes, paneles):
        for buf in (10, 20):
            f_nat = (buf / 100) / (1 + buf / 100)
            xs, ms, sds = [], [], []
            for f, carpeta in abl.items():
                m, sd, n = resumen(D[carpeta], buf, col)
                xs.append(f_nat if f == "natural" else f)
                ms.append(m)
                sds.append(sd)
                valores.append(["fig3", f"f={f}", buf, col, round(m, 2), round(sd, 2), n])
            c = COLOR_BUFFER[buf]
            ax.errorbar(xs, ms, yerr=sds, color=c, marker="o", markersize=4, linewidth=1.4,
                        elinewidth=0.7, capsize=1.8, markeredgecolor="white", markeredgewidth=0.6,
                        label=f"Buffer {buf}%", zorder=3)
            # Punto de la fracción natural: marcador hueco para distinguirlo sin texto encima.
            ax.plot(xs[0], ms[0], marker="o", markersize=5, color="white", markeredgecolor=c,
                    markeredgewidth=1.3, linestyle="none", zorder=4)
            mc, _, _ = resumen(D["concat"], buf, col)
            ax.axhline(mc, color=c, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
            valores.append(["fig3", "conventional mix", buf, col, round(mc, 2), "", ""])
        ax.set_ylabel(etiqueta)
        ax.set_xlim(0, 0.82)
        ax.set_xticks([0, 0.25, 0.5, 0.75], ["0", "0.25", "0.50", "0.75"])
    fig.supxlabel("Fraction $f$ of each batch drawn from memory", fontsize=8, y=0.07)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", markersize=5, color="white",
                              markeredgecolor=TINTA_2, markeredgewidth=1.3, linestyle="none"))
    labels.append(r"Natural fraction $f_{\mathrm{nat}}$")
    handles.append(plt.Line2D([], [], color=TINTA_2, linewidth=0.9, linestyle=(0, (3, 2))))
    labels.append("Conventional mix")
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.98),
               handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(w_pad=1.5, rect=(0, 0.04, 1, 1))
    guardar(fig, "fig3_ablacion")


def tabla(D):
    filas = [("No replay (0\\%)", D["concat"], 0), ("Conventional mix", D["concat"], 20),
             ("Balanced batches ($f=0.5$)", D["balanced"], 20), ("Adapted ER-ACE", D["erace"], 20)]
    cols = ["step4_acc_a", "step4_acc_b", "retention_a_percent", "acc_all_100_classes"]
    lineas = [
        "% Generado por paper/make_figures.py a partir de resultados/plan_gpu. No editar a mano.",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Method & $n$ & Final A & Final B & Retention A & ACC \\\\",
        "\\hline",
    ]
    for nombre, datos, buf in filas:
        celdas = []
        for c in cols:
            m, sd, n = resumen(datos, buf, c)
            celdas.append(f"${m:.1f} \\pm {sd:.1f}$")
        lineas.append(f"{nombre} & {n} & " + " & ".join(celdas) + " \\\\")
    lineas += ["\\hline", "\\end{tabular}", ""]
    (TAB / "tabla_resultados.tex").write_text("\n".join(lineas), encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    estilo()
    D = {
        "concat": leer("principal_normal"),
        "balanced": leer("principal_equilibrados"),
        "erace": leer("er_ace"),
        "natural": leer("fraccion_natural"),
        "f025": leer("fraccion_025"),
        "f075": leer("fraccion_075"),
    }
    valores: list[list] = []
    fig1(D, valores)
    fig2(D, valores)
    fig3(D, valores)
    tabla(D)
    with open(TAB / "valores_figuras.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["figura", "serie", "buffer_pct", "metrica", "media", "desv_tipica", "n"])
        w.writerows(valores)
    print("Figuras en", FIG)
    print("Tablas en", TAB)


if __name__ == "__main__":
    main()
