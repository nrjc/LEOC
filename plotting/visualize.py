import gin

from plotting.visualiser import Visualiser

gin.parse_config_file('../visualize_config.gin')
visualizer = Visualiser()
visualizer(500)
