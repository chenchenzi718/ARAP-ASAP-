from Visualization import *
from ARAP import *

if __name__ == '__main__':

    # 假设你已经有了一个用openmesh处理的网格，例如通过om.read_trimesh("path/to/your/mesh.obj")读取
    # 下面我们创建一个简单的示例网格
    om_mesh = om.read_trimesh("./example/Cow_dABF.obj")

    # 三维网格展示
    # mesh_3d_visualization(om_mesh)
    triangles = visualization_prepare(om_mesh)

    # 计算 asap 结果
    asap_solver = ASAP(om_mesh)
    v_coords = asap_solver.asap_kernel()
    # 展示 asap 结果
    mesh_2d_visualization(triangles, v_coords)

    # 计算 arap 结果并展示
    time = 3
    arap_solver = ARAP(om_mesh, time=time)
    v_coords = arap_solver.ARAP_kernel()
    mesh_2d_visualization(triangles, v_coords)
