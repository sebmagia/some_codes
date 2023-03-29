import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mpl
import os
from shapely.geometry import Point, Polygon, LineString, box
from shapely.plotting import plot_polygon, plot_points, plot_line
from shapely.figures import SIZE, BLUE, GRAY, RED, YELLOW, BLACK, set_limits
import swprepost
import matplotlib
import plotly.graph_objects as go
import pandas as pd

"""
collection of functions for generating a 3D velocity model of a site from individual 1D ground profiles.
swprepost is a very useful python package for surface wave inversion Pre- and Post-Processing
created by jpvantassel: https://github.com/jpvantassel/swprepost
"""

def create_grid(coords,xmin,xmax,ymin,ymax,wide,length):
    """
    This function creates a rectangular grid using shapely library for Polygon construction.
    Then, for each grid, identifies every ray path and its length that goes through it. 
    Rays crossing and their length is used for building merged soil profiles. 

    coords: (x1,y1,x2,y2) array with all the stations pairs. Ray paths are created connecting the pairs
    (xmin,xmax,ymin,ymax): the extent of the zone.
    wide, length: dimensions along x and y, respectively.
    """
    xx = np.hstack((coords[:, 0], coords[:, 2]))
    yy = np.hstack((coords[:, 1], coords[:, 3]))

    cols = list(np.arange(xmin-wide, xmax + wide,wide))
    rows = list(np.arange(ymin-length, ymax + length, length))

    ## create polygons. Each polygon is an object representing a grid. 
    polygons=[]
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x, y), (x + wide, y), (x + wide, y + length), (x, y + length)]))

    
    k=0

    ## predefine intersections, weights and indexes for crossing rays
    rect_isec = np.empty((len(polygons), 0)).tolist()
    weights = np.empty((len(polygons), 0)).tolist()
    indexes = np.empty((len(polygons), 0)).tolist()
    ## create line objects representing ray paths. Polygons interact with lines! The wonders of OOP
    lines=[]
    for pair in coords:
        line=LineString([pair[0:2],pair[2:4]])
        lines.append(line)

    for n, line in enumerate(lines):
        for k,rectangle in enumerate(polygons):
            bool_intersection = line.intersects(rectangle)
            intersection = line.intersection(rectangle)
            ## if the intersection is a line we save it, in the unlikely case the intersection is  a point, we are not interested.
            if bool_intersection and intersection.geom_type == 'LineString':
                rect_isec[k].append(intersection.length)
                indexes[k].append(n)
                
                print('rectangle' + str(k).zfill(2) + ' is intersected in ', intersection, ' by line '+ str(n)  )

            if bool_intersection and intersection.geom_type == 'Point':
                print('rectangle ' + str(k).zfill(2) + ' is intersected in ', intersection , ' by line '+ str(n) )


    
    c=0
    ## weights for each ray crossing each cell are defined as the ray length 
    ## divided by the sum of all ray lengths crossing the cell
    for isec, length, rectangle in zip(rect_isec,lens, polygons):
        weights[c]=[x/sum(isec) for x in isec]
        c+=1
    
    return cols, rows, rect_isec,weights, indexes, polygons




def gen_means(vels, sigmas, depths,  weights, indexes):
    """
    generate the weighted profile for each cell.
    weighting considers the profile uncertainty and ray length

    vels: all the velocity profiles
    sigmas: all the velocity profiles logarithmic standard deviation
    depth: the depth vector
    weights: all the ray length weights obtained from create_grid()
    indexes: the corresponding indexes obtained from create_grid() 

    """
    ii = 0
    vels = np.array(vels)
    for w, idx in zip(weights, indexes):
        if len(w) > 1:

            idx = np.array(idx)
            models = vels[idx]
            w = np.array(w).reshape((len(w), 1))
            weighted_model = np.sum(w * models, axis=0)
            sigmas_chosen = [sigmas[i] for i in idx]
            maxlen = np.max([len(x) for x in sigmas_chosen])
            # create an empty list to store the padded signals
            maxlen = max(len(sig) for sig in sigmas_chosen)  # find the maximum length of all signals

            arrs = np.array(
                [np.pad(sig, (0, maxlen - len(sig)), mode='constant', constant_values=1e+16) for sig in
                 sigmas_chosen]).T
            arrs = np.squeeze(arrs)
            ## weight is inverse proportional to uncertainty.
            weights_sigma = 1 / arrs
            weights_sum = np.sum(weights_sigma, axis=1)
            normalized_weights = weights_sigma / weights_sum[:, np.newaxis]

            auxiliar = [x * normalized_weights * models.T for x in w]
            weighted_model_2 = np.sum(np.sum(auxiliar, axis=0), axis=1)
            ## enforce that velocity always increases with depth. This assumption may not be always correct!
            weighted_model_2 = non_decreasing_velocity_profile(depths, weighted_model_2)[1]



            try:
                weighted_models = np.vstack((weighted_models, weighted_model))
                mean_weighted_models = np.vstack((mean_weighted_models, np.mean(weighted_model, axis=-1)))

                weighted_models_2 = np.vstack((weighted_models_2, weighted_model_2))
                mean_weighted_models_2 = np.vstack((mean_weighted_models_2, np.mean(weighted_model_2, axis=-1)))



            except ValueError:
                weighted_models = weighted_model
                mean_weighted_models = np.mean(weighted_model, axis=-1)

                weighted_models_2 = weighted_model_2
                mean_weighted_models_2 = np.mean(weighted_model_2, axis=-1)

             

        if len(w) == 1:
            idx = np.array(idx)
            weighted_model = vels[idx].reshape(vels[idx].shape[1], )
            try:
                weighted_models = np.vstack((weighted_models, weighted_model))
                mean_weighted_models = np.vstack((mean_weighted_models, np.mean(weighted_model, axis=-1)))

                weighted_models_2 = np.vstack((weighted_models_2, weighted_model))
                mean_weighted_models_2 = np.vstack((mean_weighted_models_2, np.mean(weighted_model, axis=-1)))

                

            except ValueError:
               
                weighted_models = weighted_model
                mean_weighted_models = np.mean(weighted_model, axis=-1)

                weighted_models_2 = weighted_model
                mean_weighted_models_2 = np.mean(weighted_model, axis=-1)

              

        if len(w) == 0:
            weighted_model = np.zeros(len(depths))
            weighted_model_2 = np.zeros(len(depths))
            sigma_weighted = np.zeros(len(depths))
            try:
                weighted_models = np.vstack((weighted_models, weighted_model))
                mean_weighted_models = np.vstack((mean_weighted_models, -999))

                weighted_models_2 = np.vstack((weighted_models_2, weighted_model_2))
                mean_weighted_models_2 = np.vstack((mean_weighted_models_2, -999))


         
            except UnboundLocalError:
                weighted_models = weighted_model
                mean_weighted_models = -999

                weighted_models_2 = weighted_model_2
                mean_weighted_models_2 = -999

                #sigmas_weighted = sigma_weighted
                #mean_sigmas_weighted =  20.0
        ii += 1
    return weighted_models, weighted_models_2, mean_weighted_models, mean_weighted_models_2

def finer_gm(gm):
        model = swprepost.GroundModel.from_geopsy(fname=path_gm + gm)

        ## round depths to the first decimal
        depth = np.round(model.depth, 1)
        velocity = model.vs2

        ## max depth is 9999 so redefine it to 200
        depth[-1] = 200
        ##obtain unique values of depth, depth increases always so no problem here
        depth_uni = np.unique(depth)

        ## obtain unique values of velocity, velocity may decrease with depth so we make a little fix
        indexes = np.unique(velocity, return_index=True)[1]
        vel_uni = np.array([velocity[index] for index in sorted(indexes)])
        ## define a smaller enough dz for merging vel models
        dz = 0.1
        ##
        depth_finer = np.arange(depth[0], depth[-1] + dz, dz)
        vel_finer = np.zeros(len(depth_finer))
        

        for j in range(len(vel_uni)):
            for i in range(len(depth_finer)):
                if depth_finer[i] < depth_uni[j + 1] and depth_finer[i] >= depth_uni[j]:
                    vel_finer[i] = vel_uni[j]

        return depth_finer, vel_finer

def vs3dmodel(arr, x, y, z):
    # Create a meshgrid of the x, y, and z coordinates
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Reshape the data array into a 1D array
    arr_1d = np.log10(arr.flatten())


    # Create a 3D scatter plot of the data
    fig = go.Figure(data=go.Scatter3d(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        mode='markers',
        marker=dict(
            size=5,
            color=arr_1d,
            colorscale='RdBu',
            opacity=0.7,
            colorbar=dict(
                title='Colorbar Title',
                    ),
        ),
    ))

    # Update the layout of the plot
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis Title',
            yaxis_title='Y Axis Title',
            zaxis_title='Z Axis Title',
        )
    )

    # Show the plot
    fig.show()
    data_flat = arr.ravel()
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z.ravel()
    df = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'z': z_flat,
        'data': data_flat
    })
    return df

def non_decreasing_velocity_profile(depth, velocity):
    """
    Generate a non-decreasing velocity profile. This function was done with ChatGPT hehe.

    Parameters:
        depth (array): Array of depth values.
        velocity (array): Array of velocity values.

    Returns:
        A tuple of two arrays: the modified depth and velocity arrays where the velocity profile is non-decreasing.
    """
    # Compute the cumulative maximum of the velocity array
    velocity_cummax = np.maximum.accumulate(velocity)

    # Return the modified depth and velocity arrays
    return depth, velocity_cummax
