import numpy as np
import pandas
import os
import glob
import json
import tqdm

from processing import processing_functions
from graph_generation import graph_generation_func
from analysis import analysis

def batch_analyse(spheroid_path:str,
                  intensity_path:str,
                  zRatio:float = 0.2,
                  dCells:float = 60,
                  Egg_grph:float = 1.2,
                  Egr_grph:float = 1,
                  Err_grph:float = 1.5,
                  Egg_dist:float = 1.2,
                  Egr_dist:float = 1,
                  Err_dist:float = 1.5):

    """
    
    Function specifically designed for the file storage
    system developed for the early test data.
    
    """

    resultFrame = pandas.DataFrame()

    for spheroid_name in glob.glob(os.path.join(spheroid_path, '*.csv')):

        try:

            spheroid_name = spheroid_name.split('/')[-1]
            prefix, _ = spheroid_name.split('.')
            prefix, position, date = prefix.split('_')

            intensity_name = 'intensityFrame_' + position + '_' + date + '.csv'

            spheroid_file = pandas.read_csv(os.path.join(spheroid_path,spheroid_name))
            intensity_file = pandas.read_csv(os.path.join(intensity_path,intensity_name))

            prop_frame = single_spheroid_properties(spheroid_file,
                                intensity_file,
                                zRatio = zRatio,
                                dCells = dCells,
                                Egg_grph = Egg_grph,
                                Egr_grph = Egr_grph,
                                Err_grph = Err_grph,
                                Egg_dist = Egg_dist,
                                Egr_dist = Egr_dist,
                                Err_dist = Err_dist)

            prop_frame['time'] = int(date)
            prop_frame['position'] = int(position)

            resultFrame = resultFrame.append(prop_frame)

        except Exception as e: 
            print(spheroid_name)
            print(e)
    
    return resultFrame

def single_spheroid_properties(spheroid_file:pandas.DataFrame,
                               intensity_file:pandas.DataFrame,
                               zRatio:float,
                               dCells:float,
                               Egg_grph,
                               Egr_grph,
                               Err_grph,
                               Egg_dist,
                               Egr_dist,
                               Err_dist):

    """
    
    Analysis of individual position. Function centralizes other
    analysis scripts developed.
    
    """

    
    columns = ['graph type',
               'number of nodes', 
               'average node degree',
               'degree density',
               'E_graph',
               'E_distance']

    result_frame = pandas.DataFrame(columns = columns)

    spheroid = processing_functions.single_spheroid_process(spheroid_file,
                                                            intensity_file,
                                                            include_color = True,
                                                            color_description = 'cell category')
    
    geometric_graph = graph_generation_func.generate_geometric_graph(spheroid,
                                                            zRatio = zRatio,
                                                            dCells = dCells)

    voronoi_graph = graph_generation_func.generate_voronoi_graph(spheroid,
                                                            zRatio = zRatio,
                                                            dCells = dCells)

    graphs = [geometric_graph, voronoi_graph]
    graph_types = ['geometric', 'voronoi']

    for i, g in enumerate(graphs):

        density, mean_degree = analysis.density_and_mean_degree(g)
        energy_graph = analysis.Energy_cells(g, 
                                             Egg=Egg_grph, 
                                             Egr=Egr_grph, 
                                             Err=Err_grph)
        #energy_distance = analysis.Energy_cells_distance(g,
        #                                     dCells = dCells, 
        #                                     Egg=Egg_dist, 
        #                                     Egr=Egr_dist, 
        #                                     Err=Err_dist)

        energy_distance = 0

        data = [graph_types[i], 
                len(g),
                mean_degree,
                density,
                energy_graph,
                energy_distance]
        
        loc_frame = pandas.DataFrame(data = [data], columns = columns)

        result_frame = result_frame.append(loc_frame)
                                                        
    return result_frame