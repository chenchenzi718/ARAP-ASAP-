import numpy as np

from ASAP import *


class ARAP(ASAP):
    def __init__(self, om_mesh: om.TriMesh, time):
        super(ARAP, self).__init__(om_mesh)

        # 将 asap 得到的 texcoord 结果作为 ARAP 的 ui 输入
        self.asap_kernel()

        # 记录每个三角片上的 L
        self.triangle_L = np.zeros(shape=(self.nf, 2, 2), dtype=np.float32)

        # ARAP 只固定一个点
        self.fix_point_idx = self.boundary_vertex_handle[0].idx()
        self.fix_point = np.array([0., 0.])

        # 设置迭代次数
        self.time = time

    # 此时是迭代优化，当 u 不动时，优化 L
    def matrix_L_construct(self):
        for fh in self.om_mesh.faces():
            f_idx = fh.idx()
            three_points_ = [vh.idx() for vh in self.om_mesh.fv(fh)]

            vertex_coords = self.texcoords[three_points_]

            plane = self.xy_flatten[f_idx]
            X = np.array([
                [(plane[0] - plane[1])[0], (plane[1] - plane[2])[0]],
                [(plane[0] - plane[1])[1], (plane[1] - plane[2])[1]]
            ])
            U = np.array([
                [(vertex_coords[0] - vertex_coords[1])[0], (vertex_coords[1] - vertex_coords[2])[0]],
                [(vertex_coords[0] - vertex_coords[1])[1], (vertex_coords[1] - vertex_coords[2])[1]]
            ])
            J = U @ np.linalg.inv(X)

            # 对 jacobi 矩阵进行 svd 分解
            matrix_U, diag_S, matrix_VT = np.linalg.svd(J, full_matrices=True)

            # 计算J的行列式来决定如何计算L
            if np.linalg.det(J) > 0:
                L = np.dot(matrix_U, matrix_VT)
            else:
                D = np.array([[1.0, 0.0],
                              [0.0, -1.0]])
                L = np.dot(matrix_U, np.dot(D, matrix_VT))

            self.triangle_L[f_idx] = L

    # 固定住 L 进行 u 的偏导优化
    def A_construct(self):
        a_triplet = []

        # 对 x，y 求偏导
        for vh in self.om_mesh.vertices():
            v_idx = vh.idx()
            # 取出半边的 handle
            vh_he = [heh for heh in self.om_mesh.voh(vh)]
            _sum = 0.

            if v_idx == self.fix_point_idx:
                a_triplet.append((v_idx, v_idx, 1.))
                continue

            for he in vh_he:
                tmp = 0.
                triangle = self.om_mesh.face_handle(he)
                if triangle.is_valid():
                    f_idx = triangle.idx()
                    three_points_vh_idx = [vh.idx() for vh in self.om_mesh.fv(triangle)]
                    v_index_in_triangle = three_points_vh_idx.index(v_idx)
                    tmp += self.cot[f_idx, v_index_in_triangle]

                he_pair = self.om_mesh.opposite_halfedge_handle(he)
                triangle = self.om_mesh.face_handle(he_pair)
                if triangle.is_valid():
                    f_idx = triangle.idx()
                    three_points_vh_idx = [vh.idx() for vh in self.om_mesh.fv(triangle)]
                    v_index_in_triangle = three_points_vh_idx.index(v_idx)
                    tmp += self.cot[f_idx, (v_index_in_triangle - 1) if v_index_in_triangle > 0 else 2]

                _sum += tmp

                vh_end = self.om_mesh.to_vertex_handle(he)
                vh_end_idx = vh_end.idx()
                if vh_end_idx != self.fix_point_idx:
                    a_triplet.append((v_idx, vh_end_idx, -tmp))

            a_triplet.append((v_idx, v_idx, _sum))
        return change_triplet_to_sparse_matrix(a_triplet, self.nv, self.nv)

    # 建立右侧矩阵 b
    def b_solve(self, A: coo_matrix):
        bx = np.zeros(self.nv, dtype=np.float32)
        by = np.zeros(self.nv, dtype=np.float32)

        for vh in self.om_mesh.vertices():
            vh_idx = vh.idx()
            # 取出半边的 handle
            vh_he = [heh for heh in self.om_mesh.voh(vh)]
            _sum = np.zeros(shape=(2, 1), dtype=np.float32)

            if vh_idx == self.fix_point_idx:
                bx[vh_idx] += self.fix_point[0]
                by[vh_idx] += self.fix_point[1]
                continue

            for he in vh_he:
                tmp = 0.
                triangle = self.om_mesh.face_handle(he)
                if triangle.is_valid():
                    f_idx = triangle.idx()
                    three_points_vh_idx = [vh.idx() for vh in self.om_mesh.fv(triangle)]
                    v_index_in_triangle = three_points_vh_idx.index(vh_idx)
                    tmp += self.cot[f_idx, v_index_in_triangle]

                    diff = (self.xy_flatten[f_idx, v_index_in_triangle] -
                            self.xy_flatten[f_idx, (v_index_in_triangle + 1) if v_index_in_triangle < 2 else 0])
                    right = np.array([[diff[0]], [diff[1]]])
                    _sum += self.cot[f_idx, v_index_in_triangle] * (self.triangle_L[f_idx] @ right)

                he_pair = self.om_mesh.opposite_halfedge_handle(he)
                triangle = self.om_mesh.face_handle(he_pair)
                if triangle.is_valid():
                    f_idx = triangle.idx()
                    three_points_vh_idx = [vh.idx() for vh in self.om_mesh.fv(triangle)]
                    v_index_in_triangle = three_points_vh_idx.index(vh_idx)
                    tmp += self.cot[f_idx, (v_index_in_triangle - 1) if v_index_in_triangle > 0 else 2]

                    diff = (self.xy_flatten[f_idx, v_index_in_triangle] -
                            self.xy_flatten[f_idx, (v_index_in_triangle - 1) if v_index_in_triangle > 0 else 2])
                    right = np.array([[diff[0]], [diff[1]]])
                    _sum += (self.cot[f_idx, (v_index_in_triangle - 1) if v_index_in_triangle > 0 else 2] *
                             (self.triangle_L[f_idx] @ right))

                vh_end = self.om_mesh.to_vertex_handle(he)
                vh_end_idx = vh_end.idx()
                if vh_end_idx == self.fix_point_idx:
                    bx[vh_idx] -= (-tmp) * self.fix_point[0]
                    by[vh_idx] -= (-tmp) * self.fix_point[1]

            bx[vh_idx] += _sum[0]
            by[vh_idx] += _sum[1]

        ux = spsolve(A.tocsc(), bx)
        uy = spsolve(A.tocsc(), by)

        for vh in self.om_mesh.vertices():
            vh_idx = vh.idx()
            self.texcoords[vh_idx] = [ux[vh_idx], uy[vh_idx]]

    # ARAP 主要算法
    def ARAP_kernel(self):

        A = self.A_construct()
        for i in range(self.time):
            self.matrix_L_construct()
            self.b_solve(A)
        return self.texcoords
