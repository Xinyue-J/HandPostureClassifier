from util import *


def pca_(data):
    pca = decomposition.PCA(n_components='mle', svd_solver='full')
    pca.fit(data)
    # print('explained_variance_ratio: ', pca.explained_variance_ratio_)
    # print('explained_variance: ', pca.explained_variance_)
    # print('n_components: ', pca.n_components_)

    xyz_pca = pca.fit_transform(data)
    xyz_pca = pd.DataFrame(xyz_pca)
    return xyz_pca
