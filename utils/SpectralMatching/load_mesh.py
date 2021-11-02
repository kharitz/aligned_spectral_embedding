import nibabel.freesurfer.io as fsio
import numpy as np 
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy


class LoadMesh:

    def __init__(self):
        self.weights = None
        

    def load_mesh(self,main_path,id,hemi,ne,mode,label_path):
        """
        Loads the mesh using free surfer

        path: path of the dataset
        id: patient id
        hemi: left/right hemisphere
        ne: number of eigenvectors to decompose
        mode: random walk or normalized laplacian

        returns coordinates, faces, sulcal depth, cortical thickness

        """
    
        #mesh
        path = os.path.join(main_path,id,'surf',hemi+".white")
        
        try:
            self.coords,self.faces = fsio.read_geometry(path)
            self.coords_rows = self.coords.shape[0]
            self.n = np.shape(self.coords)[0]
            
        except FileNotFoundError:
            print('Mesh File not found')

        #Sulcal depth
        path = path.replace('white','sulc')
        try:
            self.depth = fsio.read_morph_data(path)
            
        except FileNotFoundError:
            print('Depth File not found')
        
        #mesh cortical thickness
        path = path.replace('sulc','thickness')

        try:
            self.thickness = fsio.read_morph_data(path)
            
        except FileNotFoundError:
            print('Thickness File not found')

        self.mode = mode
        self.ne = ne


        path = os.path.join(label_path,id,'labels/left_cortical_surface/relabeled_labels.DKT31.manual.vtk')
        
        try:
            # self.P = fsio.read_annot(path)
            # self.P_corrected = self.P[0]+1
            # self.P = mne.read_labels_from_annot(id, parc='aparc', hemi='lh',subjects_dir=main_path)
           
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(path)
            reader.Update()

            polydata = reader.GetOutput()


            pointdata = polydata.GetPointData()
            a = vtk_to_numpy(pointdata.GetArray(0))
            a = a-1000
            a[a==-1001] = 0
            a[a==33] = 0
            a[a==32] = 0
            a[a==1] = 0
            a[a==4] = 0
           
            self.P_corrected = a

            del(reader);del(a)
            
        except FileNotFoundError:
            print('File not found label')
  