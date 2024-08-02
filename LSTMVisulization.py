import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(12, 8))

# Add rectangles for each layer
layers = [
    ("Input Layer", (0, 5)),
    ("LSTM Layer 1\n(50 units)", (1, 5)),
    ("Dropout Layer 1\n(20%)", (2, 5)),
    ("LSTM Layer 2\n(60 units)", (3, 5)),
    ("Dropout Layer 2\n(30%)", (4, 5)),
    ("LSTM Layer 3\n(80 units)", (5, 5)),
    ("Dropout Layer 3\n(40%)", (6, 5)),
    ("LSTM Layer 4\n(120 units)", (7, 5)),
    ("Dropout Layer 4\n(50%)", (8, 5)),
    ("Dense Layer\n(1 unit)", (9, 5)),
]

for layer, pos in layers:
    rect = Rectangle(pos, 1, 1, edgecolor="black", facecolor="lightblue")
    ax.add_patch(rect)
    ax.text(pos[0] + 0.5, pos[1] + 0.5, layer, 
            verticalalignment="center", horizontalalignment="center",
            fontsize=12)

# Adjust plot limits and remove axes
ax.set_xlim(-1, 11)
ax.set_ylim(4, 7)
ax.axis("off")

plt.title("LSTM Model Architecture", fontsize=16)
plt.show()
