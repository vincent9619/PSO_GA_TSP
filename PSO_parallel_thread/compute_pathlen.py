# compute the length of one path
def compute_pathlen(path, dis_mat):
    a = path[0]
    b = path[-1]
    result = dis_mat[a][b]
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        result += dis_mat[a][b]
    return result