import open3d as o3d

mesh = o3d.io.read_triangle_mesh("/home/arif-emre/Desktop/Görüntü İşleme/Aksesuar projesi/photos/gozluk.ply")
mesh.compute_vertex_normals()


try:
    o3d.visualization.draw_geometries([mesh])
except Exception as e:
    print("GUI ile açarken hata:", e)
    print("Offscreen veya OSMesa backend kullanmayı deneyin.")

