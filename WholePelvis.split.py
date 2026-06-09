import trimesh
import os
import glob
import shutil  # for copying files

# ===================== Configuration =====================
input_dir = r"J:\PG_Pelvis\WholePelvis\PCA_predict\output"
output_dir = r"J:\PG_Pelvis\WholePelvis\PCA_predict\output\split_pelvis"
opensim_geom_dir = r"C:\OpenSim 4.4\Geometry"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(opensim_geom_dir, exist_ok=True)

# Find all files ending with _pXXX.ply in the directory
ply_files = sorted(glob.glob(os.path.join(input_dir, "*_p[0-9][0-9][0-9].ply")))

if not ply_files:
    print("[WARN] No matching PLY files found")
else:
    print(f"Found {len(ply_files)} PLY files, starting splitting...")

# ===================== Iterate over each file =====================
for ply_path in ply_files:
    fname = os.path.basename(ply_path)
    # Extract _pXX part as base name
    import re
    m = re.search(r"p\d{3}", fname)
    if m:
        base_name = m.group(0)  # e.g., "p00"
    else:
        print(f"[WARN] {fname} does not match _pXX format, skipping")
        continue

    mesh = trimesh.load(ply_path)
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"[ERROR] {fname} is not a Trimesh object, skipping")
        continue

    # Unit conversion mm -> m
    # mesh.vertices /= 1000.0

    # Split into connected components
    components = mesh.split(only_watertight=False)
    if len(components) != 3:
        print(f"[WARN] {fname} detected {len(components)} connected segments, expected 3")

    # Save each segment
    seg_names = ['sac', 'pel_l', 'pel_r']
    for i, submesh in enumerate(components):
        if i >= 3:
            break  # prevent more than 3 segments
        # Unit conversion mm -> m
        submesh.vertices /= 1000.0

        out_name = f"{base_name}_{seg_names[i]}.obj"
        out_path = os.path.join(output_dir, out_name)
        submesh.export(out_path)
        print(f"Saved {fname} segment {seg_names[i]}: {out_path}")

        # Copy to OpenSim Geometry directory
        dest_path = os.path.join(opensim_geom_dir, out_name)
        shutil.copy2(out_path, dest_path)
        print(f"Copied to OpenSim Geometry: {dest_path}")
