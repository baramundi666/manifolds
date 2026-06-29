import os
import matplotlib

matplotlib.use("QtAgg", force=True)

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np
import threading
import queue


def demo():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    pts = np.random.randn(500, 3)

    state = {"idx": 0}
    q = queue.Queue()

    def redraw():
        ax.cla()

        vis = pts[:state["idx"]]

        ax.scatter(
            vis[:, 0],
            vis[:, 1],
            vis[:, 2],
            c=np.arange(len(vis)),
            cmap="plasma",
        )

        ax.set_title(f"Showing {state['idx']} points")
        fig.canvas.draw_idle()

    def next_step(event=None):
        state["idx"] = min(len(pts), state["idx"] + 1)
        redraw()

    def prev_step(event=None):
        state["idx"] = max(10, state["idx"] - 10)
        redraw()

    # Buttons
    ax_prev = plt.axes([0.72, 0.05, 0.1, 0.05])
    ax_next = plt.axes([0.84, 0.05, 0.1, 0.05])

    b_prev = Button(ax_prev, "Prev")
    b_next = Button(ax_next, "Next")

    b_prev.on_clicked(prev_step)
    b_next.on_clicked(next_step)

    # Keyboard
    def on_key(event):
        if event.key == "right":
            next_step()
        elif event.key == "left":
            prev_step()

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()

    print("Interactive window should open now.")
    print("If it does not:")
    print("  Linux: sudo apt install python3-tk")
    print("  macOS: install python.org Python")
    print("  Windows: use normal Python, not headless environment")

    plt.show(block=True)


if __name__ == "__main__":
    demo()
