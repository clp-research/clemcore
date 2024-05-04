import matplotlib.pyplot as plt
import json

def plot_grid(grid, file_path):
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])

    for y in range(len(grid)):
        for x in range(len(grid[y])):
            if grid[y][x] == "▢":
                ax.add_patch(plt.Rectangle((x, -y - 1), 1, 1, facecolor="white", edgecolor="black"))
            else:
                ax.add_patch(plt.Rectangle((x, -y - 1), 1, 1, facecolor="white", edgecolor="black"))
                ax.text(x + 0.5, -y - 0.5, grid[y][x], ha='center', va='center', color='black', fontsize=12)

    ax.autoscale_view()
    # plt.show()
    plt.savefig(file_path)


grids_path = '../../referencegame/resources/grids_v1.5.json'

data = {}
with open(grids_path) as json_file:
    data = json.load(json_file)


grid_files = {}
for type in data:

    grid_files[type] = []

    counter = 1
    for grid in data[type]:
        ascii_grid = []
        lines = grid.split('\n')
        for l in lines:
            line = l.split(' ')
            ascii_grid.append(line)

        file_path = f"grid_images/{type}_{str(counter)}.png"
        plot_grid(ascii_grid, file_path)
        grid_files[type].append(file_path)
        counter+=1

with open('grid_files.json', 'w') as f:
    json.dump(grid_files, f)