import numpy as np
import openmesh as om
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


# 创建一个处理 ASAP 网格的类
class ASAP:
    def __init__(self, om_mesh: om.TriMesh):
        self.om_mesh = om_mesh

        self.nf = self.om_mesh.n_faces()
        self.nv = self.om_mesh.n_vertices()
        self.xy_flatten = np.zeros(shape=(self.nf, 3, 2), dtype=np.float32)  # 记录等距展开在 xy 平面上的三角形
        self.cot = np.zeros(shape=(self.nf, 3), dtype=np.float32)         # 记录三角形的 cot 值
        self.diff_x = np.zeros(shape=(self.nf, 3), dtype=np.float32)      # 记录每个三角形上的 x 方向差值
        self.diff_y = np.zeros(shape=(self.nf, 3), dtype=np.float32)      # 记录每个三角形上的 y 方向差值
        self.texcoords = np.zeros((self.nv, 2), dtype=np.float32)         # 记录最终计算出来的二维坐标

        self.boundary_edge_handle = []    # 记录边界边的 handle
        self.boundary_vertex_handle = []  # 记录边界点的 handle
        self.boundary_search()

    # 将三维网格展开在二维上
    def flatten(self):
        nf = self.om_mesh.n_faces()
        # 将每个三角网格投影到 xy 平面上
        xy_plane = np.zeros(shape=(nf, 3, 2), dtype=np.float32)

        for fh in self.om_mesh.faces():
            idx = fh.idx()
            three_points_ = np.array([self.om_mesh.point(vh) for vh in self.om_mesh.fv(fh)])

            # 按照范数大小将三角形投影至平面
            xy_plane[idx, 0] = [0., 0.]
            xy_plane[idx, 1] = [np.linalg.norm(three_points_[1] - three_points_[0]), 0]

            x = (np.dot((three_points_[2] - three_points_[0]), (three_points_[1] - three_points_[0]))
                 / np.linalg.norm(three_points_[1] - three_points_[0]))
            y = np.sqrt((np.linalg.norm(three_points_[2] - three_points_[0])) ** 2 - x ** 2)
            xy_plane[idx, 2] = [x, y]

            # 记录 x 方向与 y 方向的差值
            # self.diff_x[idx] = [
            #     xy_plane[idx, 0, 0] - xy_plane[idx, 1, 0],
            #     xy_plane[idx, 1, 0] - xy_plane[idx, 2, 0],
            #     xy_plane[idx, 2, 0] - xy_plane[idx, 0, 0]
            # ]
            # self.diff_y[idx] = [
            #     xy_plane[idx, 0, 1] - xy_plane[idx, 1, 1],
            #     xy_plane[idx, 1, 1] - xy_plane[idx, 2, 1],
            #     xy_plane[idx, 2, 1] - xy_plane[idx, 0, 1]
            # ]

            self.diff_x[idx] = -np.diff(xy_plane[idx, :, 0], axis=0, append=xy_plane[idx, 0, 0])
            self.diff_y[idx] = -np.diff(xy_plane[idx, :, 1], axis=0, append=xy_plane[idx, 0, 1])

            for j in range(3):
                # 计算两个向量，对应着上面 diff 的对角
                vec1 = xy_plane[idx, j] - xy_plane[idx, (j - 1) if j > 0 else 2]
                vec2 = xy_plane[idx, (j + 1) % 3] - xy_plane[idx, (j - 1) if j > 0 else 2]
                # 计算余弦值
                cos_ij = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                # 计算余切值
                abs_cot = np.sqrt(cos_ij ** 2 / (1 - cos_ij ** 2))
                self.cot[idx, j] = abs_cot if cos_ij >= 0 else -abs_cot

        self.xy_flatten = xy_plane

    # 访问 Openmesh 边界函数并记录边界的边，点 handle
    def boundary_search(self):
        for eh in self.om_mesh.edges():
            if self.om_mesh.is_boundary(eh):
                self.boundary_edge_handle.append(eh)

        for vh in self.om_mesh.vertices():
            if self.om_mesh.is_boundary(vh):
                self.boundary_vertex_handle.append(vh)

    # 构建左侧矩阵 A，放置偏导信息；构建右侧向量 b，放置约束信息
    def equation_build(self):
        nv = self.om_mesh.n_vertices()
        nf = self.om_mesh.n_faces()
        a_triplet = []
        b = np.zeros(2 * (nv + nf), dtype=np.float32)

        # 固定住两个边界点的位置在 [0,0], [1,1] 处
        v_cons_idx_1 = self.boundary_vertex_handle[0].idx()
        v_cons_idx_2 = self.boundary_vertex_handle[len(self.boundary_vertex_handle)//2].idx()
        v_cons_1 = np.array([0., 0.])
        v_cons_2 = np.array([1., 1.])

        # 开始构建矩阵，对 a，b 求偏导
        for fh in self.om_mesh.faces():
            t = fh.idx()
            tmp = 0.
            three_points_vh = [vh for vh in self.om_mesh.fv(fh)]
            for i in range(3):
                v_index = three_points_vh[i].idx()
                tmp += self.cot[t, i] * (self.diff_x[t, i] ** 2 + self.diff_y[t, i] ** 2)

                if (v_index != v_cons_idx_1) and (v_index != v_cons_idx_2):
                    # 对 a_t 做偏导
                    a_triplet.append((2 * nv + 2 * t, 2 * v_index,
                                      -self.cot[t, i] * self.diff_x[t, i] +
                                      self.cot[t, (i - 1) if i > 0 else 2] * self.diff_x[
                                          t, (i - 1) if i > 0 else 2]))
                    a_triplet.append((2 * nv + 2 * t, 2 * v_index + 1,
                                      -self.cot[t, i] * self.diff_y[t, i] +
                                      self.cot[t, (i - 1) if i > 0 else 2] * self.diff_y[
                                          t, (i - 1) if i > 0 else 2]))

                    # 对 b_t 做偏导
                    a_triplet.append((2 * nv + 2 * t + 1, 2 * v_index,
                                      -self.cot[t, i] * self.diff_y[t, i] +
                                      self.cot[t, (i - 1) if i > 0 else 2] * self.diff_y[
                                          t, (i - 1) if i > 0 else 2]))
                    a_triplet.append((2 * nv + 2 * t + 1, 2 * v_index + 1,
                                      self.cot[t, i] * self.diff_x[t, i] -
                                      self.cot[t, (i - 1) if i > 0 else 2] * self.diff_x[
                                          t, (i - 1) if i > 0 else 2]))

                    # 保持矩阵对称
                    # 对 a_t 做偏导
                    a_triplet.append((2 * v_index, 2 * nv + 2 * t,
                                      -self.cot[t, i] * self.diff_x[t, i] +
                                      self.cot[t, (i - 1) if i > 0 else 2] * self.diff_x[
                                          t, (i - 1) if i > 0 else 2]))
                    a_triplet.append((2 * v_index + 1, 2 * nv + 2 * t,
                                      -self.cot[t, i] * self.diff_y[t, i] +
                                      self.cot[t, (i - 1) if i > 0 else 2] * self.diff_y[
                                          t, (i - 1) if i > 0 else 2]))
                    a_triplet.append((2 * v_index, 2 * nv + 2 * t + 1,
                                      -self.cot[t, i] * self.diff_y[t, i] +
                                      self.cot[t, (i - 1) if i > 0 else 2] * self.diff_y[
                                          t, (i - 1) if i > 0 else 2]))
                    a_triplet.append((2 * v_index + 1, 2 * nv + 2 * t + 1,
                                      self.cot[t, i] * self.diff_x[t, i] -
                                      self.cot[t, (i - 1) if i > 0 else 2] * self.diff_x[
                                          t, (i - 1) if i > 0 else 2]))
                elif v_index == v_cons_idx_2:
                    # 建立右侧
                    b[2 * nv + 2 * t] -= (-self.cot[t, i] * self.diff_x[t, i] +
                                            self.cot[t, (i - 1) if i > 0 else 2] * self.diff_x[
                                          t, (i - 1) if i > 0 else 2]) * v_cons_2[0]
                    b[2 * nv + 2 * t] -= (-self.cot[t, i] * self.diff_y[t, i] +
                                            self.cot[t, (i - 1) if i > 0 else 2] * self.diff_y[
                                                t, (i - 1) if i > 0 else 2]) * v_cons_2[1]
                    b[2 * nv + 2 * t + 1] -= (-self.cot[t, i] * self.diff_y[t, i] +
                                      self.cot[t, (i - 1) if i > 0 else 2] * self.diff_y[
                                          t, (i - 1) if i > 0 else 2]) * v_cons_2[0]
                    b[2 * nv + 2 * t + 1] -= (self.cot[t, i] * self.diff_x[t, i] -
                                      self.cot[t, (i - 1) if i > 0 else 2] * self.diff_x[
                                          t, (i - 1) if i > 0 else 2]) * v_cons_2[1]

            a_triplet.append((2 * nv + 2 * t, 2 * nv + 2 * t, tmp))
            a_triplet.append((2 * nv + 2 * t + 1, 2 * nv + 2 * t + 1, tmp))

        # 对 x，y 求偏导
        for vh in self.om_mesh.vertices():
            v_idx = vh.idx()
            # 取出半边的 handle
            vh_he = [heh for heh in self.om_mesh.voh(vh)]
            _sum = 0.

            if v_idx == v_cons_idx_1 or v_idx == v_cons_idx_2:
                a_triplet.append((2 * v_idx, 2 * v_idx, 1.))
                a_triplet.append((2 * v_idx + 1, 2 * v_idx + 1, 1.))
                b[2 * v_idx] = v_cons_1[0] if v_idx == v_cons_idx_1 else v_cons_2[0]
                b[2 * v_idx + 1] = v_cons_1[1] if v_idx == v_cons_idx_1 else v_cons_2[1]
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
                if vh_end_idx != v_cons_idx_1 and vh_end_idx != v_cons_idx_2:
                    a_triplet.append((2 * v_idx, 2 * vh_end_idx, -tmp))
                    a_triplet.append((2 * v_idx + 1, 2 * vh_end_idx + 1, -tmp))
                elif vh_end_idx == v_cons_idx_2:
                    b[2 * v_idx] -= (-tmp) * v_cons_2[0]
                    b[2 * v_idx + 1] -= (-tmp) * v_cons_2[1]

            a_triplet.append((2 * v_idx, 2 * v_idx, _sum))
            a_triplet.append((2 * v_idx + 1, 2 * v_idx + 1, _sum))

        return a_triplet, b, v_cons_idx_1, v_cons_idx_2, v_cons_1, v_cons_2

    # 求解最终的参数结果
    def asap_kernel(self):
        nv = self.om_mesh.n_vertices()
        nf = self.om_mesh.n_faces()

        self.flatten()
        a_triplet, b, v_cons_idx_1, v_cons_idx_2, v_cons_1, v_cons_2 = self.equation_build()

        A = change_triplet_to_sparse_matrix(a_triplet, 2 * (nv + nf), 2 * (nv + nf))
        result_coords = spsolve(A.tocsc(), b)

        v_coords = np.zeros((nv, 2), dtype=np.float32)
        for vh in self.om_mesh.vertices():
            vh_idx = vh.idx()
            if vh_idx != v_cons_idx_1 and vh_idx != v_cons_idx_2:
                v_coords[vh_idx] = [result_coords[2 * vh_idx], result_coords[2 * vh_idx + 1]]
            elif vh_idx == v_cons_idx_1:
                v_coords[vh_idx] = [v_cons_1[0], v_cons_1[1]]
            else:
                v_coords[vh_idx] = [v_cons_2[0], v_cons_2[1]]
        self.texcoords = v_coords
        return v_coords


# 将三元组转化为稀疏矩阵
def change_triplet_to_sparse_matrix(triplet, n_rows, n_cols):
    # 将三元组列表转换为行索引、列索引和值的三个数组
    rows, cols, data = zip(*triplet)

    # 将这些数组转换为NumPy数组
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)

    # 创建COO格式的稀疏矩阵
    sparse_matrix = coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    return sparse_matrix
