# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:11:20 2020

@author: yohan
"""


import numpy as np
import  matplotlib.pyplot as plt


__version__ = "0.0.1"
__author__ = "François Lauze"



def dmap(m,n,sources,pas):
     """
     Parameters
     ----------
     m : int, x-dim of the plan
     n : int, y-dim of the plan
     sources : list of points in the plan, sources with distance 0
     pas : distance between adjacent points
     Returns
     -------
     a distance map
     """
     lf = len(sources)
     nsources = np.array(sources, dtype=float)
     X = np.zeros((m,n,lf))
     x, y = np.mgrid[0:m, 0:n]
     for i in range(lf):
         xi = nsources[i,0]
         yi = nsources[i,1]
         X[:,:,i] = ((x-xi)*pas)**2 + ((y-yi)*pas)**2
     Y = np.min(X,axis=-1)
     return np.sqrt(Y)


def lemniscate_like():
     t = np.linspace(0,2*np.pi, num=100)
     x = 180*np.cos(t)*np.sin(t) +190
     y = 220*np.cos(3*t)*np.sin(t) + 225
     x.shape = (-1,1)
     y.shape = (-1,1)
     return np.hstack((x,y))
 
def patch_to_data(points,h):
    obj = points[:,0]
    dat = points[:,1:]
    resultat = []
    objectifs = []
    for i in range(len(dat)):
        for j in range(len(dat[0])):
            if obj[i] <= dat[i,j]:
                dat[i,j] = obj[i] +2*h
        resultat = np.append(resultat,np.insert(dat[i],0,h))
        objectifs = np.append(objectifs,obj[i])
    return np.reshape(resultat,(-1,13)),objectifs

def points_adjacents(liste_points,Y,m,n):
    voisins =[[0,0],[0,1],[0,-1],[1,0],[-1,0],[-1,-1],[1,1],[1,-1],[-1,1],[0, 2],[0,-2],[2,0],[-2,0]]
    xi = liste_points[:,0]
    yi = liste_points[:,1]
    pts_voisins = [evaluer_grille(Y,xi+c[0],yi+c[1],m,n) for c in voisins]
    return np.transpose(pts_voisins)

def evaluer_grille(Y,xi,yi,m,n,C=10000):
    val = []
    for k in range(len(xi)):
        if xi[k]>=0 and yi[k]>=0 and xi[k]<=m-1 and yi[k]<=n-1:
            val = np.append(val,Y[xi[k],yi[k]])
        else:
            val = np.append(val,C)
    return val

def generer_data(m,n,sources,h,nb_data,afficher):
    listes_points = np.random.randint(0,min(m,n),(nb_data,2),dtype=int)
    # We don't want points that are actual sources so we redo until it's ok
    for i,pts in enumerate(listes_points) :
        if np.any(np.equal(sources, pts).all(axis=1)) :
            while np.any(np.equal(sources, pts).all(axis=1)) : 
                pts = np.random.randint(0,min(m,n),(1,2),dtype=int)
            listes_points[i] = pts
    Y = dmap(m,n,sources,h)
    if afficher :
        plt.imshow(Y)
        plt.contour(Y, [i*20 for i in range(50)])
        plt.show()
    return patch_to_data(points_adjacents(listes_points,Y,m,n),h) 

if __name__ == "__main__":
    np.set_printoptions(linewidth=250)
    
    dat,obj = np.empty((0,13)),np.empty((0,))
    print("Création base de données")
    for k in range(10):
        print("Itérations " + str(k+1) + " sur 10")
        
        nb_data = 100  
        
        m = 500
        n = 500 
        sources = [[m-j,j] for j in np.linspace(100,400,301)]
        sources = np.reshape(sources,(301,2))
        h = np.random.random() + 0.5
        dat10,obj10 = generer_data(m,n,sources,h,nb_data,k==9)
        
        m = 500
        n = 500 
        sources = [[j,j] for j in np.linspace(100,400,301)]
        sources = np.reshape(sources,(301,2))
        h = np.random.random() + 0.5
        dat9,obj9 = generer_data(m,n,sources,h,nb_data,k==9)
        dat9 = np.append(dat10,dat9,axis=0)
        obj9 = np.append(obj10,obj9,axis=0) 
        
        m = 500
        n = 500 
        sources = [[j,250] for j in np.linspace(100,400,301)]
        sources = np.reshape(sources,(301,2))
        h = np.random.random() + 0.5
        dat8,obj8 = generer_data(m,n,sources,h,nb_data,k==9)
        dat8 = np.append(dat9,dat8,axis=0)
        obj8 = np.append(obj9,obj8,axis=0) 
        
        m = 500
        n = 500 
        sources = np.random.randint(0,min(m,n),(20,2),dtype=int)
        h = np.random.random() + 0.5
        dat7,obj7 = generer_data(m,n,sources,h,nb_data,k==9)
        dat7 = np.append(dat8,dat7,axis=0)
        obj7 = np.append(obj8,obj7,axis=0) 
        
        m = 500
        n = 500    
        sources = [[[100 + 25*i,100 + 25*j] for i in range(10)] for j in range(10)]
        sources = np.reshape(sources,(100,2))
        h = np.random.random() + 0.5
        dat6,obj6 = generer_data(m,n,sources,h,nb_data,k==9)
        dat6 = np.append(dat7,dat6,axis=0)
        obj6 = np.append(obj7,obj6,axis=0) 
        
        m = 500
        n = 500
        sources = [[250,j] for j in np.linspace(100,400,301)]
        sources = np.reshape(sources,(301,2))
        h = np.random.random() + 0.5
        dat5,obj5 = generer_data(m,n,sources,h,nb_data,k==9)
        dat5 = np.append(dat6,dat5,axis=0)
        obj5 = np.append(obj6,obj5,axis=0) 
     
        m = 500
        n = 500    
        t = np.linspace(0,2*np.pi, num=100)
        x = 100*np.cos(t) + 250
        y = 100*np.sin(t) + 250
        x.shape = (-1,1)
        y.shape = (-1,1)
        sources = np.hstack((x,y))
        sources = np.around(sources)
        h = np.random.random() + 0.5
        dat4,obj4 = generer_data(m,n,sources,h,nb_data,k==9)
        dat4 = np.append(dat5,dat4,axis=0)
        obj4 = np.append(obj5,obj4,axis=0) 
     
        m = 500
        n = 500    
        sources = ((20,77), (189,401), (257,200), (287, 455))
        h = np.random.random() + 0.5
        nb_data = 100
        dat3,obj3 = generer_data(m,n,sources,h,nb_data,k==9)
        dat3 = np.append(dat4,dat3,axis=0)
        obj3 = np.append(obj4,obj3,axis=0)  
        
        m = 500
        n = 500
        sources = [[[i,j] for i in range(100)] for j in range(10)]
        sources = np.reshape(sources,(100*10,2))
        h = np.random.random() + 0.5
        nb_data = 100
        dat2,obj2 = generer_data(m,n,sources,h,nb_data,k==9)
        dat2 = np.append(dat3,dat2,axis=0)
        obj2 = np.append(obj3,obj2,axis=0)  
        
        m = 500
        n = 500
        sources = lemniscate_like()
        sources = np.around(sources)
        h = np.random.random() + 0.5
        nb_data = 100
        dat1,obj1 = generer_data(m,n,sources,h,nb_data,k==9)
        
        dat100 = np.append(dat2,dat1,axis=0)
        obj100 = np.append(obj2,obj1,axis=0) 
        
        dat = np.append(dat,dat100,axis=0)
        obj = np.append(obj,obj100,axis=0)  
    
    #dat_scaled = copy.deepcopy(dat)
    #dat_scaled[:,1:] = (dat[:,1:] - np.tile(np.array([dat[:,1:].min(axis=1)]).transpose(), (1, 12))) / (np.tile(np.array([dat[:,1:].max(axis=1)]).transpose(), (1, 12)) - np.tile(np.array([dat[:,1:].min(axis=1)]).transpose(), (1, 12)))
    
    np.save('patch.npy', dat)
    np.save('objectifs.npy',obj)
    
    data = np.load('patch.npy')
    objectif = np.load('objectifs.npy')
