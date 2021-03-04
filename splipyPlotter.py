import click
from splipy.io import G2
import numpy as np
from mayavi import mlab

def insertUniform(X,n):
    # inserts n uniform points between each unique value in X
    X = np.unique(X)
    Y = np.zeros((n+1)*len(X)-n)
    for i in range(len(X)-1):
        idx = (n+1)*i
        Y[idx:idx+n+1] = np.linspace(X[i],X[i+1],n+1,endpoint=False)

    Y[-1] = X[-1]
    return Y

def getColor(colorID):
    if colorID == 0 or colorID == 1:
        color = [208,213,219] # polished aluminum
    elif colorID == 2:
        color = [152,215,112] # Cottrell2006iat green 1
    elif colorID == 3:
        color = [162,211,78]  # Cottrell2006iat green 2
    elif colorID == 4:
        color = [122,127,128] # metal gray
    elif colorID == 5:
        color = [132,135,137] # aluminum
    elif colorID == 6:
        color = [0,80,158] # NTNU logo color
    elif colorID == 7:
        color = [0,60,101] # SINTEF logo color
    elif colorID == 8:
        color = [0,68,123] # FFI logo color
    elif colorID == 9:
        color = [0,77,145] # background FFI color
    return np.array(color)/255

def plotGridLines(X,Ns,noKnots,d_p,linewidth=1):
    # The index of the current point in the total amount of points
    index = 0
    connections = list()
    for i in range(d_p):
        for j in range(noKnots[i]):
            # This is the tricky part: in a line, each point is connected
            # to the one following it. We have to express this with the indices
            # of the final set of points once all lines have been combined
            # together, this is why we need to keep track of the total number of
            # points already created (index)
            N = Ns[i]
            connections.append(np.vstack(
                               [np.arange(index,     index + N - 1.5),
                                np.arange(index + 1, index + N - 0.5)]).T)
            index += N


    # Now collapse all positions, scalars and connections in big arrays
    connections = np.vstack(connections)

    # Create the points
    src = mlab.pipeline.scalar_scatter(X[:,0],X[:,1],X[:,2])

    # Connect them
    src.mlab_source.dataset.lines = connections
    src.update()
    
    # Finally, display the set of lines
    mlab.pipeline.surface(src, line_width=linewidth)
    
@click.command()
@click.option('--n', type=int, default=10,help='Number of evaluations between each knot')
@click.option('--colorid', type=int, default=1,help='Coloring of model')
@click.option('--plotelementedges/--no-plotelementedges', type=bool, default=True,help='Plot element edges of the mesh')
@click.option('--plotcontrolpolygon/--no-plotcontrolpolygon', type=bool, default=True,help='Plot the control polygon')
@click.argument('infile', type=str)
def main(n,colorid,plotelementedges,plotcontrolpolygon,infile):
    g2 = G2(infile)
    patch = g2.read()
    mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    def plotSurfaceSpline(patch_i,d,d_p):
        u = insertUniform(patch_i.bases[0].knots,n)
        v = insertUniform(patch_i.bases[1].knots,n)
        XX = patch_i(u,v)
        mlab.mesh(XX[:,:,0], XX[:,:,1], XX[:,:,2], color=tuple(getColor(colorid)))
        noKnots = [len(patch_i.bases[j].knot_spans()) for j in range(2)]
        if plotelementedges:
            X = np.zeros((((noKnots[0]-1)*(n+1)+1)*noKnots[1]+((noKnots[1]-1)*(n+1)+1)*noKnots[0],d))
            # Extract values along element edges along the v-dir and u-dir
            for j in range(d):
                X[:,j] = np.append(np.reshape(XX[::(n+1),:,j],-1),np.reshape(np.transpose(XX[:,::(n+1),j]),-1))
            
            plotGridLines(X,[XX.shape[1],XX.shape[0]],noKnots,d_p)

        if plotcontrolpolygon:
            noCtrlPts = patch_i.shape
            ctrlPts = np.copy(patch_i.controlpoints)
            # Transform weighted coefficients to physical coordinates
            for j in range(d):
                ctrlPts[...,j] /= ctrlPts[...,d]

            pointsObj = mlab.mesh(ctrlPts[...,0],ctrlPts[...,1],ctrlPts[...,2],color=(0.5,0,0),representation='points')
            pointsObj.actor.property.point_size = 9.0 
            pointsObj.actor.property.render_points_as_spheres = True
            
            X = np.zeros((2*np.prod(noCtrlPts),d))
            # Extract controlpolygon along the v-dir and u-dir
            for j in range(d):
                X[:,j] = np.append(np.reshape(ctrlPts[:,:,j],-1),np.reshape(np.transpose(ctrlPts[:,:,j]),-1))

            plotGridLines(X,noCtrlPts[::-1],noCtrlPts,d_p,linewidth=0.5)

    for i in range(len(patch)):
        patch_i = patch[i]
        d = patch_i.dimension    # Dimension of the spline patch
        d_p = len(patch_i.shape) # Number of parametric directions
        if d_p == 3:
            print('Warning: Volumetric splines are not implemented, plotting surfaces instead')
            surfPatches = [patch_i.section(u=0), patch_i.section(u=-1),
                           patch_i.section(v=0), patch_i.section(v=-1),
                           patch_i.section(w=0), patch_i.section(w=-1)]

            d_p = 2
            [plotSurfaceSpline(patch_i,d,d_p) for patch_i in surfPatches]
        elif d_p == 1:
            print('Spline curves are not implemented')
            return
        else:
            plotSurfaceSpline(patch_i,d,d_p)

    mlab.show()
    
if __name__ == '__main__':
    main()
