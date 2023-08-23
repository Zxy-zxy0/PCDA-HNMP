from numpy import array
import numpy as np

cd_matrix = np.loadtxt('../Data/matrix/circRNA-Disease.txt', delimiter='\t', dtype=int)
cc_matrix = np.loadtxt('../Data/matrix/circRNA-circRNA.txt', delimiter='\t', dtype=int)
dd_matrix = np.loadtxt('../Data/matrix/Disease-Disease.txt', delimiter='\t', dtype=int)
ci_matrix = np.loadtxt('../Data/matrix/circRNA-miRNA.txt', delimiter='\t', dtype=int)
di_matrix = np.loadtxt('../Data/matrix/Disease-miRNA.txt', delimiter='\t', dtype=int)
im_matrix = np.loadtxt('../Data/matrix/miRNA-mRNA.txt', delimiter='\t', dtype=int)
md_matrix = np.loadtxt('../Data/matrix/mRNA-Disease.txt', delimiter='\t', dtype=int)

    # meta-path-1:c->c
meta_path_1 = cc_matrix
    # meta-path-2:c->c->c
meta_path_2 = np.dot(cc_matrix, cc_matrix)
    # meta-path-3:c->i->c
meta_path_3 = np.dot(ci_matrix, ci_matrix.T)
    # meta-path-4:c->d->c
meta_path_4 = np.dot(cd_matrix, cd_matrix.T)
    # meta-path-5:d->d
meta_path_5 = dd_matrix
    # meta-path-6:d->d->d
meta_path_6 = np.dot(dd_matrix, dd_matrix)
    # meta-path-7:d->i->d
meta_path_7 = np.dot(di_matrix, di_matrix.T)
    # meta-path-8:d->c->d
meta_path_8 = np.dot(cd_matrix.T, cd_matrix)
    # meta-path-9:d->m->d
meta_path_9 = np.dot(md_matrix.T, md_matrix)
    # meta-path-10:c->i
meta_path_10 = ci_matrix
    # meta-path-11:c->c->i
meta_path_11 = np.dot(cc_matrix, ci_matrix)
    # meta-path-12:c->d->i
meta_path_12 = np.dot(cd_matrix, di_matrix)
    # meta-path-13:i->d
meta_path_13 = di_matrix.T
    # meta-path-14:i->d->d
meta_path_14 = np.dot(di_matrix.T, dd_matrix)
    # meta-path-15:i->c->d
meta_path_15 = np.dot(ci_matrix.T, cd_matrix)
    # meta-path-16:i->m->d
meta_path_16 = np.dot(im_matrix, md_matrix)
    # meta-path-17:i->d->i
meta_path_17 = np.dot(di_matrix.T, di_matrix)
    # meta-path-18:i->c->i
meta_path_18 = np.dot(ci_matrix.T, ci_matrix)
    # meta-path-19:m->d->m
meta_path_19 = np.dot(md_matrix, md_matrix.T)
    # meta-path-20:m->i->m
meta_path_20 = np.dot(im_matrix.T, im_matrix)
# meta-path-21:i->m->i
meta_path_21 = np.dot(im_matrix, im_matrix.T)

cd1 = np.where(meta_path_1 != 0)
np.savetxt('../Data/meta_path/circRNA-circRNA/cc.txt', array(cd1).T, fmt='%d')
cd2 = np.where(meta_path_2 != 0)
np.savetxt('../Data/meta_path/circRNA-circRNA/ccc.txt', array(cd2).T, fmt='%d')
cd3 = np.where(meta_path_3 != 0)
np.savetxt('../Data/meta_path/circRNA-circRNA/cic.txt', array(cd3).T, fmt='%d')
cd4 = np.where(meta_path_4 != 0)
np.savetxt('../Data/meta_path/circRNA-circRNA/cdc.txt', array(cd4).T, fmt='%d')
cd5 = np.where(meta_path_5 != 0)
np.savetxt('../Data/meta_path/Disease-Disease/dd.txt', array(cd5).T, fmt='%d')
cd6 = np.where(meta_path_6 != 0)
np.savetxt('../Data/meta_path/Disease-Disease/ddd.txt', array(cd6).T, fmt='%d')
cd7 = np.where(meta_path_7 != 0)
np.savetxt('../Data/meta_path/Disease-Disease/did.txt', array(cd7).T, fmt='%d')
cd8 = np.where(meta_path_8 != 0)
np.savetxt('../Data/meta_path/Disease-Disease/dcd.txt', array(cd8).T, fmt='%d')
cd9 = np.where(meta_path_9 != 0)
np.savetxt('../Data/meta_path/Disease-Disease/dmd.txt', array(cd9).T, fmt='%d')
cd10 = np.where(meta_path_10 != 0)
np.savetxt('../Data/meta_path/circRNA-miRNA/ci.txt', array(cd10).T, fmt='%d')
cd11 = np.where(meta_path_11 != 0)
np.savetxt('../Data/meta_path/circRNA-miRNA/cci.txt', array(cd11).T, fmt='%d')
cd12 = np.where(meta_path_12 != 0)
np.savetxt('../Data/meta_path/circRNA-miRNA/cdi.txt', array(cd12).T, fmt='%d')


