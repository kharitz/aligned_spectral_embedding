
import mne
from mne.transforms import apply_trans
from mne.io.constants import FIFF


import numpy
import mayavi
import mayavi.mlab as mlab
from mayavi.mlab import triangular_mesh,points3d



def plot_graph(rr_mm,tris,vectors,num,eigen=False):

    import numpy
    import mayavi
    import mayavi.mlab as mlab
    from mayavi.mlab import triangular_mesh,points3d

    mlab.figure()

    if eigen:
        triangular_mesh(rr_mm[:,0],rr_mm[:,1],rr_mm[:,2],tris, scalars = vectors[:,num],colormap='jet')
    else:
        triangular_mesh(rr_mm[:,0],rr_mm[:,1],rr_mm[:,2],tris, scalars = vectors,colormap='jet')
   
    

def plot(rr_mm,tris):
    renderer = mne.viz.backends.renderer.create_3d_figure(
        size = (600,600),bgcolor = 'w',scene = False)
    gray = (0.5,0.5,0.5)

    renderer.mesh(*rr_mm.T,triangles = tris,color = gray)
    view_kwargs = dict(elevation=90,azimuth = 0)
    mne.viz.set_3d_view(
        figure = renderer.figure,distance=350,focalpoint=(0.,0.,40.),
        **view_kwargs
    )

    renderer.show()

    