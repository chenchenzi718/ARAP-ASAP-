实现采用 python 3.9.19（openmesh 在 python 里只有 python 3.9 支持），
openmesh 1.2.1，
numpy 1.26.4，
scipy 1.12.0
matplotlib 3.8.0，用来显示二维参数化结果
pyvista 0.43.4. 用来显示输入的三维网格图样  
    （pyvista 可能存在与 numpy 的兼容性问题，代码中暂时将这部分注释了，想运行可以将 main.py 里的 mesh_3d_visualization(om_mesh) 不注释
    ，同时 Visualization.py 中的 “import pyvista as pv” 不注释，函数 mesh_3d_visualization 的定义不注释即可）。

直接在 main.py 内点击运行即可输出 ASAP，ARAP 参数化结果. 总共两张图输出，先出来的是 ASAP，ARAP会跑一段时间后出现