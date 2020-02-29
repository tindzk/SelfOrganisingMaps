import h5py
import numpy as np
from PIL import Image

def getPic(fname):
    img = Image.open(fname)
    img.thumbnail((100, 100), Image.ANTIALIAS)
    p = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    p = np.mean(p,axis=2)/255.
    p = np.flipud(p)
    return p.T.flatten()

P = np.array([],dtype=float)
P = np.hstack([P, getPic('images/hannes.jpg')])
#P = np.hstack([P, getPic('images/stuart.png')])
P = np.hstack([P, getPic('images/laura.png')])
P = np.hstack([P, getPic('images/luke.jpg')])
P = np.hstack([P, getPic('images/rodrigo.png')])
#P = np.hstack([P, getPic('images/mitch.png')])
#P = np.hstack([P, getPic('images/tony.png')])

h5f = h5py.File('testPatterns.h5','w')
h5f.create_dataset('P', data=P)
h5f.close()
