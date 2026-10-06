import trimesh

stl_path = r"Y:\Máster\TFM\Imágenes STL\3DTeethSeg\Imágenes\lower\OSZV3Q38\OSZV3Q38_lower.stl"
obj_path = r"Y:\Máster\TFM\Imágenes STL\3DTeethSeg\Imágenes\lower\OSZV3Q38\OSZV3Q38_lower.obj"
mesh = trimesh.load(obj_path)
mesh.export(stl_path)  # Lo guarda como STL