import math
from datetime import datetime
from nicegui import ui

def drawLinePlot(ui):
    line_plot = ui.line_plot(n=2, limit=20, figsize=(3, 2), update_every=5) \
        .with_legend(['sin', 'cos'], loc='upper center', ncol=2)

    def update_line_plot() -> None:
        now = datetime.now()
        x = now.timestamp()
        y1 = math.sin(x)
        y2 = math.cos(x)
        line_plot.push([now], [[y1], [y2]])

    line_updates = ui.timer(0.1, update_line_plot, active=False)
    line_checkbox = ui.checkbox('active').bind_value(line_updates, 'active')
