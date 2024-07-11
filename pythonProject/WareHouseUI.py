import tkinter as tk
from tkinter import messagebox
import random
import csv
import torch
from Model.DL_Network_Model import Net
from Model.Gane_Agent import game_agent

# Global variables for counters
win_count = 0
loss_count = 0
tie_count = 0

# Load the model
file_name_model_latest_version = 'Model/model_latest_version.pt'
model_latest_version = Net()
state_dict_latest_version = torch.load(file_name_model_latest_version)['state_dict']
model_latest_version.load_state_dict(state_dict_latest_version)

# Create main window
root = tk.Tk()
root.title("Warehouse Management")

# Warehouse dimensions
rows, cols = 3, 3

# Create a 2D array to store the state of the cells
warehouse = [[None for _ in range(cols)] for _ in range(rows)]

# Function to update the color of a cell based on its value
def update_button_color(button, value):
    if value == -1:
        button.config(bg='green', highlightbackground='green')
    elif value == 1:
        button.config(bg='beige', highlightbackground='beige')
    elif value == 2:
        button.config(bg='red', highlightbackground='red')
    else:
        button.config(bg='gray', highlightbackground='gray')

# Function to add an item to the warehouse
def add_item():
    global win_count, loss_count, tie_count
    quantity = int(entry_quantity.get())
    value = int(entry_value.get())
    arr = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    torch_tensor = torch.tensor(value, dtype=torch.float).unsqueeze(0)
    next_vision = model_latest_version(torch_tensor)
    game_a = game_agent(torch_tensor, 0)
    for k in range(9):
        if torch_tensor[k] != 0:
            next_vision[k] = -1.1
    next_step = next_vision.argmax().item()
    print(next_vision)

    # Simulate game result (for example purpose)
    game_result = random.choice(["Win", "Loss", "Tie"])  # Replace with actual game result logic
    if game_result == "Win":
        win_count += 1
    elif game_result == "Loss":
        loss_count += 1
    else:
        tie_count += 1

    # Update labels
    win_label.config(text=f"Wins: {win_count}")
    loss_label.config(text=f"Losses: {loss_count}")
    tie_label.config(text=f"Ties: {tie_count}")

    return next_step

# Function to reset the warehouse to its initial state
def reset_warehouse():
    for i in range(rows):
        for j in range(cols):
            warehouse[i][j] = None
            update_button_color(buttons[i][j], None)

# Function to randomize items in the warehouse
def randomize_items():
    reset_warehouse()
    values = [-1, 1, 2]
    counts = {-1: min(5, 2 * cols), 1: min(5, 3 * cols), 2: min(5, 2 * cols)}

    for value in values:
        row_ranges = {-1: range(1, -1, -1), 1: range(4, 1, -1), 2: range(6, 4, -1)}
        row_range = row_ranges[value]
        remaining_count = counts[value]

        positions = [(i, j) for i in row_range for j in range(cols) if warehouse[i][j] is None]
        random.shuffle(positions)

        for i, j in positions:
            if remaining_count > 0:
                warehouse[i][j] = value
                update_button_color(buttons[i][j], value)
                remaining_count -= 1
            else:
                break

# Function to export warehouse data to CSV
def export_warehouse():
    with open('warehouse_export.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in warehouse:
            writer.writerow(row)
    messagebox.showinfo("Export", "Warehouse data exported to warehouse_export.csv")

# Function to remove the closest item from the bottom up
def remove_closest_item():
    closest_item = None
    closest_position = None

    for i in range(rows - 1, -1, -1):
        for j in range(cols):
            if warehouse[i][j] is not None:
                closest_item = warehouse[i][j]
                closest_position = (i, j)
                break
        if closest_position:
            break

    if closest_position:
        i, j = closest_position
        warehouse[i][j] = None
        update_button_color(buttons[i][j], None)
        messagebox.showinfo("Remove Item", f"Removed item with value {closest_item} from position ({i}, {j})")
    else:
        messagebox.showinfo("Remove Item", "No items to remove")

# Create warehouse buttons
warehouse_frame = tk.Frame(root)
warehouse_frame.grid(row=0, column=0, padx=10, pady=10)

buttons = [[None for _ in range(cols)] for _ in range(rows)]
for i in range(rows):
    for j in range(cols):
        button = tk.Button(warehouse_frame, width=4, height=2, bg='gray', state=tk.NORMAL)
        button.grid(row=i, column=j)
        buttons[i][j] = button

# Create input fields for quantity and value
input_frame = tk.Frame(root)
input_frame.grid(row=1, column=0, padx=10, pady=10)

tk.Label(input_frame, text="Quantity:").grid(row=0, column=0, padx=5, pady=5)
entry_quantity = tk.Entry(input_frame)
entry_quantity.grid(row=0, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Value:").grid(row=0, column=2, padx=5, pady=5)
entry_value = tk.Entry(input_frame)
entry_value.grid(row=0, column=3, padx=5, pady=5)

tk.Button(input_frame, text="Add Item", command=add_item).grid(row=0, column=4, padx=5, pady=5)
tk.Button(input_frame, text="Reset", command=reset_warehouse).grid(row=0, column=5, padx=5, pady=5)
tk.Button(input_frame, text="Random", command=randomize_items).grid(row=0, column=6, padx=5, pady=5)
tk.Button(input_frame, text="Export", command=export_warehouse).grid(row=0, column=7, padx=5, pady=5)
tk.Button(input_frame, text="Remove Closest", command=remove_closest_item).grid(row=0, column=8, padx=5, pady=5)

# Create labels to display win, loss, and tie counters
counter_frame = tk.Frame(root)
counter_frame.grid(row=2, column=0, padx=10, pady=10)

win_label = tk.Label(counter_frame, text=f"Wins: {win_count}")
win_label.grid(row=0, column=0, padx=5, pady=5)
loss_label = tk.Label(counter_frame, text=f"Losses: {loss_count}")
loss_label.grid(row=0, column=1, padx=5, pady=5)
tie_label = tk.Label(counter_frame, text=f"Ties: {tie_count}")
tie_label.grid(row=0, column=2, padx=5, pady=5)

root.mainloop()
