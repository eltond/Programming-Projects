import glob
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint

def CreateList(Locale):
    ListOfFiles = []
    for name in glob.glob(Locale):
        ListOfFiles.append(name)
    return ListOfFiles

def Run_PCA(f):
    FigFile = f[:-3]+"png"
    if FigFile.find("Tok") == -1:
        col = "blue"
    else:
        col = "red"
    source = pd.read_csv(f)
    newsource = source - source.mean()
    datacov = newsource.cov()
    eig_val_cov, eig_vec_cov = np.linalg.eig(datacov)
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    eig_pairs.sort()
    eig_pairs.reverse()
    matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
    transformed = newsource.as_matrix().dot(matrix_w)
    plt.plot(transformed[0:len(source),0],transformed[0:len(source),1],\
         'o', markersize=7, color=col, alpha=0.5, label='class1')  
    return ((eig_pairs[0]+eig_pairs[1]), f[f.find("armb"):])   

def main():
    Images = []
    MatrixDict = {}
    MatrixDict['File'] = []
    MatrixDict['Eigenvalue1'] = []
    MatrixDict['Eigenvector1'] = [] 
    MatrixDict['Eigenvalue2'] = []
    MatrixDict['Eigenvector2'] = []    
    path = "C:\Users\NYU\SeaFlow2\*"
    FileList = CreateList(path)
    for i in FileList:
        Images.append(Run_PCA(i))
    for i,j in enumerate(Images):
        MatrixDict['File'].append(Images[i][1])
        MatrixDict['Eigenvalue1'].append(Images[i][0][0])
        MatrixDict['Eigenvector1'].append(Images[i][0][1]) 
        MatrixDict['Eigenvalue2'].append(Images[i][0][2])
        MatrixDict['Eigenvector2'].append(Images[i][0][3])
    FinalMatrix = pd.DataFrame(MatrixDict, columns=['File','Eigenvalue1','Eigenvalue2','Eigenvector1','Eigenvector2'])
    print FinalMatrix


if __name__ == "__main__":
    main()




    
