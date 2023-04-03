import os
import imageio
from settings import *

network_name = 'two_d_lattice_union_Erdos_Renyi'
# 'two_d_lattice_union_diagonals'
# 'two_d_lattice_union_Erdos_Renyi'

# Set input and output paths
input_path = './data/visualizing-spread/videos/output'
output_path = './data/visualizing-spread/videos/' + network_name + '_'+str(network_size)+'.mp4'

fps = 1
macro_block_size = 16
writer = imageio.get_writer(output_path, fps=fps, macro_block_size=macro_block_size)
file_count = len(os.listdir(input_path))

for i in range(file_count):
    # Load image and resize to a multiple of the macro block size
    image = imageio.imread(os.path.join(input_path, str(i) + ".png"))
    height, width, _ = image.shape
    height = (height // macro_block_size) * macro_block_size
    width = (width // macro_block_size) * macro_block_size
    image = image[:height, :width, :]

    # Add resized image to video
    writer.append_data(image)

writer.close()



