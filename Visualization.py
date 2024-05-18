# import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import openmesh as om


# 显示3维网格数据
# def mesh_3d_visualization(om_mesh):
#     # 获取numpy版本
#     np_version = np.__version__
#     # 比较版本号，如果numpy版本大于等于1.20
#     if int(np_version.split('.')[0]) > 1 or (
#             int(np_version.split('.')[0]) == 1 and int(np_version.split('.')[1]) >= 20):
#         np.bool = np.bool_  # 用np.bool_替代np.bool
#
#     # 提取顶点坐标
#     verts = np.array([om_mesh.point(vh) for vh in om_mesh.vertices()], dtype=np.float32)
#
#     # 提取面信息并构建PyVista所需格式
#     faces = np.hstack([[3] + [vh.idx() for vh in om_mesh.fv(fh)] for fh in om_mesh.faces()])
#
#     # 创建PyVista网格
#     pv_mesh = pv.PolyData(verts, faces)
#
#     # 使用PyVista显示网格
#     plotter = pv.Plotter()
#     plotter.add_mesh(pv_mesh, color='white', show_edges=True)
#     plotter.show()


# 显示二维网格数据
def mesh_2d_visualization(triangles, vertices):
    # 假设vertices是一个二维数组，其中包含了网格所有顶点的(x, y)坐标
    # 例如：vertices = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 1.5]])
    # vertices = np.array([
    #     [0, 0],  # Vertex 0
    #     [1, 0],  # Vertex 1
    #     [0, 1],  # Vertex 2
    #     [1, 1],  # Vertex 3
    #     [0.5, 1.5]  # Vertex 4
    # ])

    # 假设triangles是一个包含三角形顶点ID的数组
    # 例如：triangles = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    # triangles = np.array([
    #     [0, 1, 2],  # Triangle 0
    #     [1, 3, 2],  # Triangle 1
    #     [2, 3, 4]  # Triangle 2
    # ])

    # 创建一个matplotlib的Triangulation对象
    triangulation = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)

    # 绘制网格
    plt.figure()
    plt.triplot(triangulation, '-', color='black', linewidth=0.2)
    plt.gca().set_aspect('equal')
    plt.show()


# 构建 openmesh 的面到点的索引，如下所示：
#     triangles = np.array([                 vertices = np.array([
#         [0, 1, 2],  # Triangle 0                  [0, 0],  # Vertex 0
#         [1, 3, 2],  # Triangle 1                  [1, 0],  # Vertex 1
#         [2, 3, 4]  # Triangle 2                   [0.5, 1.5]  # Vertex 4
#     ])
def visualization_prepare(om_mesh: om.TriMesh):
    # 初始化一个列表来收集所有三角形的顶点索引
    triangles_list = []

    # 遍历网格中的每个面
    for fh in om_mesh.faces():
        # 初始化一个列表来收集当前面的顶点索引
        face_vertices = []
        # 遍历面上的每个顶点
        for vh in om_mesh.fv(fh):
            # 收集顶点索引
            face_vertices.append(vh.idx())
        # 将当前面的顶点索引列表添加到三角形列表中
        triangles_list.append(face_vertices)

    # 将列表转换为numpy数组
    triangles = np.array(triangles_list)
    return triangles
