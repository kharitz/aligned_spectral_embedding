

main_path = '/data/datasets/Mindboggle101/processed/processed_FreeSurfer/' #MRI scans
label_path = '/data/datasets/Mindboggle101/processed/labels_manual/' #manual labels
file_save_path = '/Aligned_files'

# main_path = 'D:/MITACS/MindBoggle' #MRI scans
# label_path = 'D:/MITACS/MindBoggle/labels_manual'
# file_save_path = 'D:/MITACS/Aligned_files'

hemi= ['lh']
ne= 5
mode='randomwalk'

krot = 5                    #number of eigen vectos used for transformation
matching_samples = 5000     #  number of points used to find transformation #5000
matching_mode = 'partial'   #partial or complete
niter = 4*krot              #no iterations to refine basis transformation matrix (sign flip, reordering)
ki = 5                      #eigen vectors used for matching
w_sulcal = 1                #sulcal weights

plot = False