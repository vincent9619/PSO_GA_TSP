 # compute the length of current particles
import compute_pathlen as cpl
def compute_paths(paths, dis_mat):
    result = []
    for one in paths:
        length = cpl.compute_pathlen(one, dis_mat)
        result.append(length)
    return result