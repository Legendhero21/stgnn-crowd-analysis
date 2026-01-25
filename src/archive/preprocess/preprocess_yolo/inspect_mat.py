import scipy.io as sio
import numpy as np

mat_path = r"D:/stgnn_project/data/mat_dataset/ground_truth/GT_img002296.mat"

mat = sio.loadmat(mat_path)

print("TOP-LEVEL KEYS:", [k for k in mat.keys() if not k.startswith("__")])

info = mat.get("image_info", None)
print("\nimage_info type:", type(info))
print("image_info shape:", getattr(info, "shape", None))

if info is not None:
    print("\nimage_info[0] type:", type(info[0]))
    print("image_info[0][0] type:", type(info[0][0]))
    try:
        print("image_info[0][0] dtype:", getattr(info[0][0], "dtype", None))
    except Exception as e:
        print("dtype error:", e)

    # try common access patterns, but just print shapes
    for path, arr in [
        ("image_info[0][0]['location']",  info[0][0]["location"]  if isinstance(info[0][0], np.void) and "location" in info[0][0].dtype.names else None),
    ]:
        if arr is not None:
            print(f"\n{path} type:", type(arr))
            print(f"{path} shape:", getattr(arr, "shape", None))
