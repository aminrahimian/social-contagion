import imageio
from visualizing_spread import *
from settings import root_visualizing_spread_address, simulator_ID
# Set input and output paths
input_path = visualizing_spread_output_address
output_path = root_visualizing_spread_address + simulator_ID + '/' + network_name + '_'+str(network_size)+'.mp4'
if highlight_infecting_edges:
    output_path = root_visualizing_spread_address + simulator_ID + '/' + network_name + '_with_dynamic_red_edges_' + str(network_size) + '.mp4'

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



