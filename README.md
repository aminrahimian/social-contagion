# social contagion

code to reproduce simulations and plots for our paper entitled, Long ties accelerate noisy threshold-based contagions, available here: http://web.mit.edu/rahimian/www/EMRS_2018_long.pdf

## getting started

### software

+ R 3.5.1
+ Python 3.6

*Depending on the size of the dataset and parameters simulations, user might require access to high-performance computing (HPC)*

## simulations execution:

### Settings:

+ **settings.py:** define the params,network_group, spreading_models, simulation settings, modes settings and etc.

+ **models.py:** define the spread models and params

### Computaion

+ **computing_spread_time_c1_c2_interpolation.py**: Python file to compute spread times in c1_c2_interpolation and store the results in pickled files.

+ **computing_spread_time_c1_union_ER_w_delta.py**: Python file to compute spread times in c1_union_ER with different delta nd store the results in pickled files.

+ **computing_spread_time_c1_union_ER_w_rho.py**: Python file to compute spread times in c1_union_ER with different rho and store the results in pickled files.

+ **computing_spread_time_ck_union_ER.py**: Python file to compute spread times in c1_union_ER and store the results in pickled files.

+ **computing_spread_time_lattice_union_ER.py**: Python file to compute spread times in lattice_union_ER and store the results in pickled files.

+ **computing_spread_time_lattice_union_ER_vs_network_size.py**: Python file to compute spread times in lattice_union_ER with different network size and store the results in pickled files.

+ **measuring_spread_time_real_networks.py**: Python file to measure spread times in real networks and store the results in pickled files.

+ **measuring_strcutural_properties_real_networks.py**: Python file to measure strcutural properties in real networks and store the results in pickled files.

### Data Dumping

+ **dump_properties_data.py**: Python file to dump strcutural properties in real networks using outputs for measuring_strcutural_properties_real_networks.py

+ **dump_spreading_data.py**: Python file to dump spreading data in real networks using outputs for measuring_spread_time_real_networks.py

+ **dump_spread_time_c_k_union_ER.py**: Python file to dump spreading data in c_k_union_ER using outputs for computing_spread_time_ck_union_ER.py

### Visualization

+ **plot_Quantile_fractional_series.R**: R script to generate pdf files for quantile fractional using outputs for dump_spreading_data.py

+ **plot_activation_functions.py**: Python file to generate pdf files for activation functions

+ **plot_boxed_group_mean_average_clustering.R**: R script to generate pdf files for average clustering using outputs for dump_properties_data.py

+ **plot_boxed_group_mean_spreading_times.R**: R script to generate pdf files for spreading times using outputs for dump_spreading_data.py

+ **plot_chami_networks.R**: R script to generate pdf files for spreading times using outputs for dump_spreading_data.py

+ **plot_square_layout_graphs.py**: Python file to generate pdf files of simulation of square layout *layout = 'lattice'*.

+ **plot_circular_layout_graphs.py**: Python file to generate pdf files of simulation of circular layout *layout = 'circular'*.

+ **plot_empirical_adoption_rates.R**: R script to generate pdf files for empirical adoption rates

+ **plot_group_mean_spreading_times_by_intervention_size.R**: R script to generate pdf files for spreading times using outputs for dump_spreading_data.py

+ **plot_group_mean_spreading_times_by_network_size.R**: R script to generate pdf files for spreading times using outputs for dump_spreading_data.py

+ **plot_properties_banerjee.R**: R script to generate pdf files for properties of banerjee dataset using outputs for dump_properties_data.py

+ **plot_spread_time_c_k_union_ER.R**: R script to generate pdf files for spreading times using outputs for dump_spread_time_c_k_union_ER.py

+ **plot_spread_time_c1_c2_interpolation.py**: Python file to generate pdf files for spreading times using outputs for computing_spread_time_c1_c2_interpolation.py

+ **plot_spread_time_c1_union_ER_w_delta.py**: Python file to generate pdf files for spreading times using outputs for computing_spread_time_c1_union_ER_w_delta.py

+ **plot_spread_time_histograms.py**: Python file to generate pdf files for spreading times using outputs for measuring_spread_time_real_networks.py

+ **plot_spread_time_lattice_union_ER.py**: Python file to generate pdf files for spreading times using outputs for computing_spread_time_lattice_union_ER.py

+ **plot_spread_time_lattice_union_ER_vs_network_size.R**: Python file to generate pdf files for spreading times using outputs for computing_spread_time_lattice_union_ER_vs_network_size.py

+ **plot_spreading_times_banerjee.R**: R script to generate pdf files for spreading times using outputs for dump_spreading_data.py

+ **plot_spreading_times_cai.R**: R script to generate pdf files for spreading times using outputs for dump_spreading_data.py

+ **plot_spreading_times_chami.R**: R script to generate pdf files for spreading times using outputs for dump_spreading_data.py

+ **plot_spreading_times_chami_union.R**: R script to generate pdf files for spreading times using outputs for dump_spreading_data.py

+ **plot_spreading_times_fb.R**: R script to generate pdf files for spreading times using outputs for dump_spreading_data.py

+ **plot_structural_properties_histograms.py**: Python file to generate pdf files for histograms of structural properties using outputs for measuring_strcutural_properties_real_networks.py

+ **ploting_spread_time_c1_union_ER_w_rho.py**: Python file to generate pdf files for spreading times using outputs for computing_spread_time_c1_union_ER_w_rho.py

+ **video_generating.py**: Python file to generate video files for simulation of spreading using outputs for visualizing_spread.py

+ **visualizing_spread.py**: Python file to generate video files for simulation of spreading using outputs for itself

