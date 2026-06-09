# ============================================================
# DESCRIPTION: PCA shape model registration for pelvis only with Hip Joint Center estimation
# ============================================================

import os
import copy
import glob
import logging
import numpy as np
from gias3.applications.general import init_log
from gias3.learning import PCA
from gias3.mesh import vtktools
from gias3.registration import shapemodel
from gias3.musculoskeletal import mocap_landmark_preprocess
from gias3.fieldwork.field import geometric_field
from gias3.musculoskeletal import fw_model_landmarks
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)
r2c = shapemodel.r2c13
FTOL = 1e-6

mean_mesh_path = r"J:\PG_Pelvis\WholePelvis\pca_meshes_mean\PGpelvis_mean.ply"
ssm_path = r"J:\PG_Pelvis\WholePelvis\pca_meshes_mean\PGpelvis.pc"
landmark_nodes = r"J:\PG_Pelvis\WholePelvis\PCA_predict\input\TemplateMesh_landmarkVertices.txt"
landmark_targets_dir = r"J:\PG_Pelvis\WholePelvis\PCA_predict\input"
out_path = r"J:\PG_Pelvis\WholePelvis\PCA_predict\output\PCApredict_wholePelvis.ply"


fit_comps = [0, 1, 2, 3, 4, 5, 6, 7]
fit_mode = "ts"
mw = 0.1  
fit_scale = False
auto_align = True
view = True
points_only = False
marker_radius = 5.0
skin_pad = 5.0

# ===================== Utility Functions =========================
def _load_node_indices(path):
    arr = []
    with open(path, 'r') as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            try:
                arr.append(int(parts[0]))
            except ValueError:
                try:
                    arr.append(int(parts[-1]))
                except Exception:
                    raise ValueError(f"Cannot parse node index line: {ln}")
    return np.array(arr, dtype=int)


def _load_landmark_targets(path):
    coords = []
    with open(path, 'r') as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            floats = []
            for p in parts:
                try:
                    floats.append(float(p))
                    if len(floats) == 3:
                        break
                except:
                    continue
            if len(floats) != 3:
                raise ValueError(f"Cannot parse landmark coordinates: {ln}")
            coords.append(floats)  
    return np.array(coords, dtype=float)


def _make_ldmk_evaluator_from_indices(indices):
    indices = np.array(indices, dtype=int)

    def evaluator(flat_coords):
        arr = np.asarray(flat_coords)
        if arr.ndim == 1:
            pts = arr.reshape((-1, 3))
        elif arr.ndim == 2:
            if arr.shape[1] == 3:
                pts = arr
            else:
                pts = arr.reshape((-1, 3))
        else:
            raise ValueError("ldmk_evaluator: unsupported input format")
        return pts[indices, :]

    return evaluator

def _auto_rigid_init(source_pts, target_pts, fit_scale=False):
    source_pts = np.asarray(source_pts)
    target_pts = np.asarray(target_pts)
    if source_pts.shape[0] != target_pts.shape[0]:
        raise ValueError("source_pts and target_pts must have the same number of points")

    centroid_source = source_pts.mean(axis=0)
    centroid_target = target_pts.mean(axis=0)

    src_centered = source_pts - centroid_source
    tgt_centered = target_pts - centroid_target

    if fit_scale:
        scale = np.linalg.norm(tgt_centered) / np.linalg.norm(src_centered)
        src_centered *= scale
    else:
        scale = 1.0

    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_target - R @ centroid_source

    import scipy.spatial.transform
    r = scipy.spatial.transform.Rotation.from_matrix(R)
    rx, ry, rz = r.as_euler('xyz', degrees=False)
    return np.hstack([t, [rx, ry, rz]])

def preprocess_pelvis_landmarks(target_landmarks_dict, marker_radius=5.0, skin_pad=5.0):
    pelvis_landmarks = [
        'pelvis-LASIS', 'pelvis-RASIS', 'pelvis-LPSIS', 'pelvis-RPSIS', 'pelvis-Sacral'
    ]
    preprocessor = mocap_landmark_preprocess.preprocessors['pelvis']
    coords = [target_landmarks_dict[n] for n in pelvis_landmarks]
    try:
        new_coords = preprocessor(marker_radius, skin_pad, *coords)
    except mocap_landmark_preprocess.InsufficientLandmarksError:
        log.warning("Insufficient landmarks for pelvis preprocessing, using original coordinates")
        new_coords = coords
    processed = {n: new_coords[i] for i, n in enumerate(pelvis_landmarks)}
    return processed


# ===================== Main Function =====================
def run_registration():
    mean_mesh = vtktools.loadpoly(mean_mesh_path)
    mean_pts = mean_mesh.v

    prefix = os.path.splitext(os.path.basename(landmark_targets))[0]  # "p12"

    if os.path.exists(landmark_nodes) and os.path.exists(landmark_targets):
        landmark_indices = _load_node_indices(landmark_nodes)
        raw_targets = _load_landmark_targets(landmark_targets)

        if len(landmark_indices) < raw_targets.shape[0]:
            raise ValueError(
                f"Number of landmark_nodes {len(landmark_indices)} < number of landmark_targets {raw_targets.shape[0]}")
        elif len(landmark_indices) > raw_targets.shape[0]:
            log.warning(
                f"Number of landmark_nodes {len(landmark_indices)} > number of landmark_targets {raw_targets.shape[0]}, will truncate sequentially")
            landmark_indices = landmark_indices[:raw_targets.shape[0]]

        vertex_indices = landmark_indices
        ldmk_evaluator = _make_ldmk_evaluator_from_indices(vertex_indices)

        landmark_names = ['pelvis-LASIS', 'pelvis-RASIS', 'pelvis-LPSIS', 'pelvis-RPSIS', 'pelvis-Sacral']
        raw_dict = {name: raw_targets[i] for i, name in enumerate(landmark_names)}
        processed_dict = preprocess_pelvis_landmarks(raw_dict, marker_radius, skin_pad)

        template_pts = mean_pts[vertex_indices, :]
        target_pts = np.array([processed_dict[n] for n in landmark_names])

        ldmk_targs = target_pts

        ply_landmark_path = rf"J:\PG_Pelvis\WholePelvis\PCA_predict\output\pelvis_landmarks_{prefix}.ply"
        writer = vtktools.Writer(v=ldmk_targs, f=np.zeros((0, 3), dtype=int))
        writer.write(ply_landmark_path)
        print(f"✅ Processed landmarks saved: {ply_landmark_path}")

    else:
        log.info("No landmark files found, continuing without landmarks.")
        return

    init_rigid = _auto_rigid_init(template_pts, ldmk_targs, fit_scale=fit_scale)
    t0 = init_rigid.copy()

    ssm = PCA.loadPrincipalComponents(ssm_path)
    n_modes = len(fit_comps)
    pc_weights0 = np.zeros(n_modes, dtype=float)
    x0 = np.hstack([pc_weights0, t0])

    def objective(x):
        pc_weights = x[:n_modes]
        t = x[n_modes:n_modes + 3]
        r = x[n_modes + 3:]
        recon_flat = ssm.mean.copy() + np.dot(ssm.modes[:, fit_comps], pc_weights)
        recon_pts = recon_flat.reshape((-1, 3))
        import scipy.spatial.transform
        R = scipy.spatial.transform.Rotation.from_euler('xyz', r).as_matrix()
        recon_pts = (R @ recon_pts.T).T + t
        recon_ldmk_pts = ldmk_evaluator(recon_pts)
        dist2 = np.sum((recon_ldmk_pts - ldmk_targs) ** 2)
        reg = mw * np.sum(pc_weights ** 2)
        return dist2 + reg

    from scipy.optimize import minimize
    res = minimize(objective, x0, method='L-BFGS-B', options={'ftol': FTOL, 'maxiter': 2000})

    xopt = res.x
    pc_weights_opt = xopt[:n_modes]
    t_opt = xopt[n_modes:n_modes + 3]
    r_opt = xopt[n_modes + 3:]
    final_flat = ssm.mean.copy() + np.dot(ssm.modes[:, fit_comps], pc_weights_opt)
    final_pts = final_flat.reshape((-1, 3))
    import scipy.spatial.transform
    R_opt = scipy.spatial.transform.Rotation.from_euler('xyz', r_opt).as_matrix()
    final_pts = (R_opt @ final_pts.T).T + t_opt


    vertex_indices = _load_node_indices(landmark_nodes)

    landmark_names = [
        'pelvis-RASIS', 'pelvis-LASIS',
        'pelvis-RPSIS', 'pelvis-LPSIS',
    ]

    name_to_index = {name: i for i, name in enumerate(landmark_names)}

    def get_landmark(name):
        idx = name_to_index[name]
        vertex_id = vertex_indices[idx]
        return final_pts[vertex_id]

    P_RASIS = get_landmark('pelvis-RASIS')
    P_LASIS = get_landmark('pelvis-LASIS')
    P_RPSIS = get_landmark('pelvis-RPSIS')
    P_LPSIS = get_landmark('pelvis-LPSIS')

    ASIS_mid = 0.5 * (P_RASIS + P_LASIS)
    PSIS_mid = 0.5 * (P_RPSIS + P_LPSIS)


    # ---- z axis: LASIS -> RASIS ----
    z_axis = P_RASIS - P_LASIS
    z_axis /= np.linalg.norm(z_axis)

    # ---- x axis: posterior negative ----
    # anterior positive
    x_axis = ASIS_mid - PSIS_mid
    x_axis /= np.linalg.norm(x_axis)

    # ---- y axis: right-hand system ----
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # ---- re-orthogonalize x ----
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # ---- rotation matrix ----
    # columns = axes
    R_pelvis = np.column_stack([x_axis, y_axis, z_axis])

    # ==================== transform mesh ====================

    # translate to origin
    final_pts = final_pts - origin

    # rotate into pelvis ACS
    final_pts = (R_pelvis.T @ final_pts.T).T

    print("[INFO] Pelvis ACS alignment applied.")


    reg_mesh = copy.deepcopy(mean_mesh)
    reg_mesh.v = final_pts
    reg_mesh.name = "pelvis_mesh"

    out_path_prefixed = rf"J:\PG_Pelvis\WholePelvis\PCA_predict\output\PCApredict_wholePelvis_{prefix}.ply"
    writer = vtktools.Writer(v=final_pts, f=reg_mesh.f)
    writer.write(out_path_prefixed)
    print(f"✅ Registration mesh saved: {out_path_prefixed}")

    vertex_indices = _load_node_indices(landmark_nodes)
    landmark_nodes_new_coords = final_pts[vertex_indices, :]
    landmark_nodes_orig_coords = mean_pts[vertex_indices, :]
    landmark_nodes_displacement = landmark_nodes_new_coords - landmark_nodes_orig_coords

    landmark_names = [
        'pelvis-RASIS', 'pelvis-LASIS', 'pelvis-RPSIS', 'pelvis-LPSIS', 'pelvis-Sacral',
        'SIJ-L1', 'SIJ-L2', 'SIJ-R1', 'SIJ-R2',
        'pubic-L1', 'pubic-R1',
        'L5-S1',
        'ACE-l-group-1', 'ACE-l-group-2', 'ACE-l-group-3', 'ACE-l-group-4', 'ACE-l-group-5', 'ACE-l-group-6',
        'ACE-l-group-7', 'ACE-l-group-8', 'ACE-l-group-9', 'ACE-l-group-10', 'ACE-l-group-11', 'ACE-l-group-12',
        'ACE-l-group-13', 'ACE-l-group-14', 'ACE-l-group-15', 'ACE-l-group-16', 'ACE-l-group-17', 'ACE-l-group-18',
        'ACE-l-group-19', 'ACE-l-group-20', 'ACE-l-group-21', 'ACE-l-group-22', 'ACE-l-group-23', 'ACE-l-group-24',
        'ACE-l-group-25', 'ACE-l-group-26', 'ACE-l-group-27', 'ACE-l-group-28', 'ACE-l-group-29', 'ACE-l-group-30',
        'ACE-l-group-31', 'ACE-l-group-32', 'ACE-l-group-33', 'ACE-l-group-34', 'ACE-l-group-35', 'ACE-l-group-36',
        'ACE-l-group-37', 'ACE-l-group-38', 'ACE-l-group-39', 'ACE-l-group-40', 'ACE-l-group-41', 'ACE-l-group-42',
        'ACE-l-group-43', 'ACE-l-group-44', 'ACE-l-group-45', 'ACE-l-group-46', 'ACE-l-group-47', 'ACE-l-group-48',
        'ACE-l-group-49', 'ACE-l-group-50', 'ACE-l-group-51', 'ACE-l-group-52', 'ACE-l-group-53', 'ACE-l-group-54',
        'ACE-l-group-55', 'ACE-l-group-56', 'ACE-l-group-57', 'ACE-l-group-58', 'ACE-l-group-59', 'ACE-l-group-60',
        'ACE-l-group-61', 'ACE-l-group-62', 'ACE-l-group-63', 'ACE-l-group-64', 'ACE-l-group-65', 'ACE-l-group-66',
        'ACE-l-group-67', 'ACE-l-group-68', 'ACE-l-group-69', 'ACE-l-group-70', 'ACE-l-group-71', 'ACE-l-group-72',
        'ACE-l-group-73', 'ACE-l-group-74', 'ACE-l-group-75', 'ACE-l-group-76', 'ACE-l-group-77', 'ACE-l-group-78',
        'ACE-l-group-79', 'ACE-l-group-80', 'ACE-l-group-81', 'ACE-l-group-82', 'ACE-l-group-83', 'ACE-l-group-84',
        'ACE-l-group-85', 'ACE-l-group-86', 'ACE-l-group-87', 'ACE-l-group-88', 'ACE-l-group-89', 'ACE-l-group-90',
        'ACE-l-group-91', 'ACE-l-group-92', 'ACE-l-group-93', 'ACE-l-group-94', 'ACE-l-group-95', 'ACE-l-group-96',
        'ACE-l-group-97', 'ACE-l-group-98', 'ACE-l-group-99', 'ACE-l-group-100', 'ACE-l-group-101', 'ACE-l-group-102',
        'ACE-l-group-103', 'ACE-l-group-104', 'ACE-l-group-105', 'ACE-l-group-106', 'ACE-l-group-107',
        'ACE-l-group-108', 'ACE-l-group-109', 'ACE-l-group-110', 'ACE-l-group-111', 'ACE-l-group-112',
        'ACE-l-group-113', 'ACE-l-group-114', 'ACE-l-group-115', 'ACE-l-group-116', 'ACE-l-group-117',
        'ACE-l-group-118', 'ACE-l-group-119', 'ACE-l-group-120', 'ACE-l-group-121', 'ACE-l-group-122',
        'ACE-l-group-123', 'ACE-l-group-124', 'ACE-l-group-125', 'ACE-l-group-126', 'ACE-l-group-127',
        'ACE-l-group-128', 'ACE-l-group-129', 'ACE-l-group-130', 'ACE-l-group-131', 'ACE-l-group-132',
        'ACE-l-group-133', 'ACE-l-group-134', 'ACE-l-group-135', 'ACE-l-group-136', 'ACE-l-group-137',
        'ACE-l-group-138', 'ACE-l-group-139', 'ACE-l-group-140', 'ACE-l-group-141', 'ACE-l-group-142',
        'ACE-l-group-143', 'ACE-l-group-144', 'ACE-l-group-145', 'ACE-l-group-146', 'ACE-l-group-147',
        'ACE-l-group-148', 'ACE-l-group-149', 'ACE-l-group-150', 'ACE-l-group-151', 'ACE-l-group-152',
        'ACE-l-group-153', 'ACE-l-group-154', 'ACE-l-group-155', 'ACE-l-group-156', 'ACE-l-group-157',
        'ACE-l-group-158', 'ACE-l-group-159', 'ACE-l-group-160', 'ACE-l-group-161', 'ACE-l-group-162',
        'ACE-l-group-163', 'ACE-l-group-164', 'ACE-l-group-165', 'ACE-l-group-166', 'ACE-l-group-167',
        'ACE-l-group-168', 'ACE-l-group-169', 'ACE-l-group-170', 'ACE-l-group-171', 'ACE-l-group-172',
        'ACE-l-group-173', 'ACE-l-group-174', 'ACE-l-group-175', 'ACE-l-group-176', 'ACE-l-group-177',
        'ACE-l-group-178', 'ACE-l-group-179', 'ACE-l-group-180', 'ACE-l-group-181', 'ACE-l-group-182',
        'ACE-l-group-183', 'ACE-l-group-184', 'ACE-l-group-185', 'ACE-l-group-186', 'ACE-l-group-187',
        'ACE-l-group-188', 'ACE-l-group-189', 'ACE-l-group-190', 'ACE-l-group-191', 'ACE-l-group-192',
        'ACE-l-group-193', 'ACE-l-group-194', 'ACE-l-group-195', 'ACE-l-group-196', 'ACE-l-group-197',
        'ACE-l-group-198', 'ACE-l-group-199', 'ACE-l-group-200', 'ACE-l-group-201', 'ACE-l-group-202',
        'ACE-l-group-203', 'ACE-l-group-204', 'ACE-l-group-205', 'ACE-l-group-206', 'ACE-l-group-207',
        'ACE-l-group-208', 'ACE-l-group-209', 'ACE-l-group-210', 'ACE-l-group-211', 'ACE-l-group-212',
        'ACE-l-group-213', 'ACE-l-group-214', 'ACE-l-group-215', 'ACE-l-group-216', 'ACE-l-group-217',
        'ACE-l-group-218', 'ACE-l-group-219', 'ACE-l-group-220', 'ACE-l-group-221', 'ACE-l-group-222',
        'ACE-l-group-223', 'ACE-l-group-224', 'ACE-l-group-225', 'ACE-l-group-226', 'ACE-l-group-227',
        'ACE-l-group-228', 'ACE-l-group-229', 'ACE-l-group-230', 'ACE-l-group-231', 'ACE-l-group-232',
        'ACE-l-group-233', 'ACE-l-group-234', 'ACE-l-group-235', 'ACE-l-group-236', 'ACE-l-group-237',
        'ACE-l-group-238', 'ACE-l-group-239', 'ACE-l-group-240', 'ACE-l-group-241', 'ACE-l-group-242',
        'ACE-l-group-243', 'ACE-l-group-244', 'ACE-l-group-245', 'ACE-l-group-246', 'ACE-l-group-247',
        'ACE-l-group-248', 'ACE-l-group-249', 'ACE-l-group-250', 'ACE-l-group-251', 'ACE-l-group-252',
        'ACE-l-group-253', 'ACE-l-group-254', 'ACE-l-group-255', 'ACE-l-group-256', 'ACE-l-group-257',
        'ACE-l-group-258', 'ACE-l-group-259', 'ACE-l-group-260', 'ACE-l-group-261', 'ACE-l-group-262',
        'ACE-l-group-263', 'ACE-l-group-264', 'ACE-l-group-265', 'ACE-l-group-266', 'ACE-l-group-267',
        'ACE-l-group-268', 'ACE-l-group-269', 'ACE-l-group-270', 'ACE-l-group-271', 'ACE-l-group-272',
        'ACE-l-group-273', 'ACE-l-group-274', 'ACE-l-group-275', 'ACE-l-group-276', 'ACE-l-group-277',
        'ACE-l-group-278', 'ACE-l-group-279', 'ACE-l-group-280', 'ACE-l-group-281', 'ACE-l-group-282',
        'ACE-l-group-283', 'ACE-l-group-284', 'ACE-l-group-285', 'ACE-l-group-286', 'ACE-l-group-287',
        'ACE-l-group-288', 'ACE-l-group-289', 'ACE-l-group-290', 'ACE-l-group-291', 'ACE-l-group-292',
        'ACE-l-group-293', 'ACE-l-group-294', 'ACE-l-group-295', 'ACE-l-group-296', 'ACE-l-group-297',
        'ACE-l-group-298', 'ACE-l-group-299', 'ACE-l-group-300', 'ACE-l-group-301', 'ACE-l-group-302',
        'ACE-l-group-303', 'ACE-l-group-304', 'ACE-l-group-305', 'ACE-l-group-306', 'ACE-l-group-307',
        'ACE-l-group-308', 'ACE-l-group-309', 'ACE-l-group-310', 'ACE-l-group-311', 'ACE-l-group-312',
        'ACE-l-group-313', 'ACE-l-group-314', 'ACE-l-group-315', 'ACE-l-group-316', 'ACE-l-group-317',
        'ACE-l-group-318', 'ACE-l-group-319', 'ACE-l-group-320', 'ACE-l-group-321', 'ACE-l-group-322',
        'ACE-l-group-323', 'ACE-l-group-324', 'ACE-l-group-325', 'ACE-l-group-326', 'ACE-l-group-327',
        'ACE-l-group-328', 'ACE-l-group-329', 'ACE-l-group-330', 'ACE-l-group-331', 'ACE-l-group-332',
        'ACE-l-group-333', 'ACE-l-group-334', 'ACE-l-group-335', 'ACE-l-group-336', 'ACE-l-group-337',
        'ACE-l-group-338', 'ACE-l-group-339', 'ACE-l-group-340', 'ACE-l-group-341', 'ACE-l-group-342',
        'ACE-l-group-343', 'ACE-l-group-344', 'ACE-l-group-345', 'ACE-l-group-346', 'ACE-l-group-347',
        'ACE-l-group-348', 'ACE-l-group-349', 'ACE-l-group-350', 'ACE-l-group-351', 'ACE-l-group-352',
        'ACE-l-group-353', 'ACE-l-group-354', 'ACE-l-group-355', 'ACE-l-group-356', 'ACE-l-group-357',
        'ACE-l-group-358', 'ACE-l-group-359', 'ACE-l-group-360', 'ACE-l-group-361', 'ACE-l-group-362',
        'ACE-l-group-363', 'ACE-l-group-364', 'ACE-l-group-365', 'ACE-l-group-366', 'ACE-l-group-367',
        'ACE-l-group-368', 'ACE-l-group-369', 'ACE-l-group-370', 'ACE-l-group-371', 'ACE-l-group-372',
        'ACE-l-group-373', 'ACE-l-group-374', 'ACE-l-group-375', 'ACE-l-group-376', 'ACE-l-group-377',
        'ACE-l-group-378', 'ACE-l-group-379', 'ACE-l-group-380', 'ACE-l-group-381', 'ACE-l-group-382',
        'ACE-l-group-383', 'ACE-l-group-384', 'ACE-l-group-385', 'ACE-l-group-386', 'ACE-l-group-387',
        'ACE-l-group-388', 'ACE-l-group-389', 'ACE-l-group-390', 'ACE-l-group-391', 'ACE-l-group-392',
        'ACE-l-group-393', 'ACE-l-group-394', 'ACE-l-group-395', 'ACE-l-group-396', 'ACE-l-group-397',
        'ACE-l-group-398', 'ACE-l-group-399', 'ACE-l-group-400', 'ACE-l-group-401', 'ACE-l-group-402',
        'ACE-l-group-403', 'ACE-l-group-404', 'ACE-l-group-405', 'ACE-l-group-406', 'ACE-l-group-407',
        'ACE-l-group-408', 'ACE-l-group-409', 'ACE-l-group-410', 'ACE-l-group-411', 'ACE-l-group-412',
        'ACE-l-group-413', 'ACE-l-group-414', 'ACE-l-group-415', 'ACE-l-group-416', 'ACE-l-group-417',
        'ACE-l-group-418', 'ACE-l-group-419', 'ACE-l-group-420', 'ACE-l-group-421', 'ACE-l-group-422',
        'ACE-l-group-423', 'ACE-l-group-424', 'ACE-l-group-425', 'ACE-l-group-426', 'ACE-l-group-427',
        'ACE-l-group-428', 'ACE-l-group-429', 'ACE-l-group-430', 'ACE-l-group-431', 'ACE-l-group-432',
        'ACE-l-group-433', 'ACE-l-group-434', 'ACE-l-group-435', 'ACE-l-group-436', 'ACE-l-group-437',
        'ACE-l-group-438', 'ACE-l-group-439', 'ACE-l-group-440', 'ACE-l-group-441', 'ACE-l-group-442',
        'ACE-l-group-443', 'ACE-l-group-444', 'ACE-l-group-445', 'ACE-l-group-446', 'ACE-l-group-447',
        'ACE-l-group-448', 'ACE-l-group-449', 'ACE-l-group-450', 'ACE-l-group-451', 'ACE-l-group-452',
        'ACE-l-group-453', 'ACE-l-group-454', 'ACE-l-group-455', 'ACE-l-group-456', 'ACE-l-group-457',
        'ACE-l-group-458', 'ACE-l-group-459', 'ACE-l-group-460', 'ACE-l-group-461', 'ACE-l-group-462',
        'ACE-l-group-463', 'ACE-l-group-464', 'ACE-l-group-465', 'ACE-l-group-466', 'ACE-l-group-467',
        'ACE-l-group-468', 'ACE-l-group-469', 'ACE-l-group-470', 'ACE-l-group-471', 'ACE-l-group-472',
        'ACE-l-group-473', 'ACE-l-group-474', 'ACE-l-group-475', 'ACE-l-group-476', 'ACE-l-group-477',
        'ACE-l-group-478', 'ACE-l-group-479', 'ACE-l-group-480', 'ACE-l-group-481', 'ACE-l-group-482',
        'ACE-l-group-483', 'ACE-l-group-484', 'ACE-l-group-485', 'ACE-l-group-486', 'ACE-l-group-487',
        'ACE-l-group-488', 'ACE-l-group-489', 'ACE-l-group-490', 'ACE-l-group-491',

        'ACE-r-group-1', 'ACE-r-group-2', 'ACE-r-group-3',
        'ACE-r-group-4', 'ACE-r-group-5', 'ACE-r-group-6', 'ACE-r-group-7', 'ACE-r-group-8', 'ACE-r-group-9',
        'ACE-r-group-10', 'ACE-r-group-11', 'ACE-r-group-12', 'ACE-r-group-13', 'ACE-r-group-14', 'ACE-r-group-15',
        'ACE-r-group-16', 'ACE-r-group-17', 'ACE-r-group-18', 'ACE-r-group-19', 'ACE-r-group-20', 'ACE-r-group-21',
        'ACE-r-group-22', 'ACE-r-group-23', 'ACE-r-group-24', 'ACE-r-group-25', 'ACE-r-group-26', 'ACE-r-group-27',
        'ACE-r-group-28', 'ACE-r-group-29', 'ACE-r-group-30', 'ACE-r-group-31', 'ACE-r-group-32', 'ACE-r-group-33',
        'ACE-r-group-34', 'ACE-r-group-35', 'ACE-r-group-36', 'ACE-r-group-37', 'ACE-r-group-38', 'ACE-r-group-39',
        'ACE-r-group-40', 'ACE-r-group-41', 'ACE-r-group-42', 'ACE-r-group-43', 'ACE-r-group-44', 'ACE-r-group-45',
        'ACE-r-group-46', 'ACE-r-group-47', 'ACE-r-group-48', 'ACE-r-group-49', 'ACE-r-group-50', 'ACE-r-group-51',
        'ACE-r-group-52', 'ACE-r-group-53', 'ACE-r-group-54', 'ACE-r-group-55', 'ACE-r-group-56', 'ACE-r-group-57',
        'ACE-r-group-58', 'ACE-r-group-59', 'ACE-r-group-60', 'ACE-r-group-61', 'ACE-r-group-62', 'ACE-r-group-63',
        'ACE-r-group-64', 'ACE-r-group-65', 'ACE-r-group-66', 'ACE-r-group-67', 'ACE-r-group-68', 'ACE-r-group-69',
        'ACE-r-group-70', 'ACE-r-group-71', 'ACE-r-group-72', 'ACE-r-group-73', 'ACE-r-group-74', 'ACE-r-group-75',
        'ACE-r-group-76', 'ACE-r-group-77', 'ACE-r-group-78', 'ACE-r-group-79', 'ACE-r-group-80', 'ACE-r-group-81',
        'ACE-r-group-82', 'ACE-r-group-83', 'ACE-r-group-84', 'ACE-r-group-85', 'ACE-r-group-86', 'ACE-r-group-87',
        'ACE-r-group-88', 'ACE-r-group-89', 'ACE-r-group-90', 'ACE-r-group-91', 'ACE-r-group-92', 'ACE-r-group-93',
        'ACE-r-group-94', 'ACE-r-group-95', 'ACE-r-group-96', 'ACE-r-group-97', 'ACE-r-group-98', 'ACE-r-group-99',
        'ACE-r-group-100', 'ACE-r-group-101', 'ACE-r-group-102', 'ACE-r-group-103', 'ACE-r-group-104',
        'ACE-r-group-105', 'ACE-r-group-106', 'ACE-r-group-107', 'ACE-r-group-108', 'ACE-r-group-109',
        'ACE-r-group-110', 'ACE-r-group-111', 'ACE-r-group-112', 'ACE-r-group-113', 'ACE-r-group-114',
        'ACE-r-group-115', 'ACE-r-group-116', 'ACE-r-group-117', 'ACE-r-group-118', 'ACE-r-group-119',
        'ACE-r-group-120', 'ACE-r-group-121', 'ACE-r-group-122', 'ACE-r-group-123', 'ACE-r-group-124',
        'ACE-r-group-125', 'ACE-r-group-126', 'ACE-r-group-127', 'ACE-r-group-128', 'ACE-r-group-129',
        'ACE-r-group-130', 'ACE-r-group-131', 'ACE-r-group-132', 'ACE-r-group-133', 'ACE-r-group-134',
        'ACE-r-group-135', 'ACE-r-group-136', 'ACE-r-group-137', 'ACE-r-group-138', 'ACE-r-group-139',
        'ACE-r-group-140', 'ACE-r-group-141', 'ACE-r-group-142', 'ACE-r-group-143', 'ACE-r-group-144',
        'ACE-r-group-145', 'ACE-r-group-146', 'ACE-r-group-147', 'ACE-r-group-148', 'ACE-r-group-149',
        'ACE-r-group-150', 'ACE-r-group-151', 'ACE-r-group-152', 'ACE-r-group-153', 'ACE-r-group-154',
        'ACE-r-group-155', 'ACE-r-group-156', 'ACE-r-group-157', 'ACE-r-group-158', 'ACE-r-group-159',
        'ACE-r-group-160', 'ACE-r-group-161', 'ACE-r-group-162', 'ACE-r-group-163', 'ACE-r-group-164',
        'ACE-r-group-165', 'ACE-r-group-166', 'ACE-r-group-167', 'ACE-r-group-168', 'ACE-r-group-169',
        'ACE-r-group-170', 'ACE-r-group-171', 'ACE-r-group-172', 'ACE-r-group-173', 'ACE-r-group-174',
        'ACE-r-group-175', 'ACE-r-group-176', 'ACE-r-group-177', 'ACE-r-group-178', 'ACE-r-group-179',
        'ACE-r-group-180', 'ACE-r-group-181', 'ACE-r-group-182', 'ACE-r-group-183', 'ACE-r-group-184',
        'ACE-r-group-185', 'ACE-r-group-186', 'ACE-r-group-187', 'ACE-r-group-188', 'ACE-r-group-189',
        'ACE-r-group-190', 'ACE-r-group-191', 'ACE-r-group-192', 'ACE-r-group-193', 'ACE-r-group-194',
        'ACE-r-group-195', 'ACE-r-group-196', 'ACE-r-group-197', 'ACE-r-group-198', 'ACE-r-group-199',
        'ACE-r-group-200', 'ACE-r-group-201', 'ACE-r-group-202', 'ACE-r-group-203', 'ACE-r-group-204',
        'ACE-r-group-205', 'ACE-r-group-206', 'ACE-r-group-207', 'ACE-r-group-208', 'ACE-r-group-209',
        'ACE-r-group-210', 'ACE-r-group-211', 'ACE-r-group-212', 'ACE-r-group-213', 'ACE-r-group-214',
        'ACE-r-group-215', 'ACE-r-group-216', 'ACE-r-group-217', 'ACE-r-group-218', 'ACE-r-group-219',
        'ACE-r-group-220', 'ACE-r-group-221', 'ACE-r-group-222', 'ACE-r-group-223', 'ACE-r-group-224',
        'ACE-r-group-225', 'ACE-r-group-226', 'ACE-r-group-227', 'ACE-r-group-228', 'ACE-r-group-229',
        'ACE-r-group-230', 'ACE-r-group-231', 'ACE-r-group-232', 'ACE-r-group-233', 'ACE-r-group-234',
        'ACE-r-group-235', 'ACE-r-group-236', 'ACE-r-group-237', 'ACE-r-group-238', 'ACE-r-group-239',
        'ACE-r-group-240', 'ACE-r-group-241', 'ACE-r-group-242', 'ACE-r-group-243', 'ACE-r-group-244',
        'ACE-r-group-245', 'ACE-r-group-246', 'ACE-r-group-247', 'ACE-r-group-248', 'ACE-r-group-249',
        'ACE-r-group-250', 'ACE-r-group-251', 'ACE-r-group-252', 'ACE-r-group-253', 'ACE-r-group-254',
        'ACE-r-group-255', 'ACE-r-group-256', 'ACE-r-group-257', 'ACE-r-group-258', 'ACE-r-group-259',
        'ACE-r-group-260', 'ACE-r-group-261', 'ACE-r-group-262', 'ACE-r-group-263', 'ACE-r-group-264',
        'ACE-r-group-265', 'ACE-r-group-266', 'ACE-r-group-267', 'ACE-r-group-268', 'ACE-r-group-269',
        'ACE-r-group-270', 'ACE-r-group-271', 'ACE-r-group-272', 'ACE-r-group-273', 'ACE-r-group-274',
        'ACE-r-group-275', 'ACE-r-group-276', 'ACE-r-group-277', 'ACE-r-group-278', 'ACE-r-group-279',
        'ACE-r-group-280', 'ACE-r-group-281', 'ACE-r-group-282', 'ACE-r-group-283', 'ACE-r-group-284',
        'ACE-r-group-285', 'ACE-r-group-286', 'ACE-r-group-287', 'ACE-r-group-288', 'ACE-r-group-289',
        'ACE-r-group-290', 'ACE-r-group-291', 'ACE-r-group-292', 'ACE-r-group-293', 'ACE-r-group-294',
        'ACE-r-group-295', 'ACE-r-group-296', 'ACE-r-group-297', 'ACE-r-group-298', 'ACE-r-group-299',
        'ACE-r-group-300', 'ACE-r-group-301', 'ACE-r-group-302', 'ACE-r-group-303', 'ACE-r-group-304',
        'ACE-r-group-305', 'ACE-r-group-306', 'ACE-r-group-307', 'ACE-r-group-308', 'ACE-r-group-309',
        'ACE-r-group-310', 'ACE-r-group-311', 'ACE-r-group-312', 'ACE-r-group-313', 'ACE-r-group-314',
        'ACE-r-group-315', 'ACE-r-group-316', 'ACE-r-group-317', 'ACE-r-group-318', 'ACE-r-group-319',
        'ACE-r-group-320', 'ACE-r-group-321', 'ACE-r-group-322', 'ACE-r-group-323', 'ACE-r-group-324',
        'ACE-r-group-325', 'ACE-r-group-326', 'ACE-r-group-327', 'ACE-r-group-328', 'ACE-r-group-329',
        'ACE-r-group-330', 'ACE-r-group-331', 'ACE-r-group-332', 'ACE-r-group-333', 'ACE-r-group-334',
        'ACE-r-group-335', 'ACE-r-group-336', 'ACE-r-group-337', 'ACE-r-group-338', 'ACE-r-group-339',
        'ACE-r-group-340', 'ACE-r-group-341', 'ACE-r-group-342', 'ACE-r-group-343', 'ACE-r-group-344',
        'ACE-r-group-345', 'ACE-r-group-346', 'ACE-r-group-347', 'ACE-r-group-348', 'ACE-r-group-349',
        'ACE-r-group-350', 'ACE-r-group-351', 'ACE-r-group-352', 'ACE-r-group-353', 'ACE-r-group-354',
        'ACE-r-group-355', 'ACE-r-group-356', 'ACE-r-group-357', 'ACE-r-group-358', 'ACE-r-group-359',
        'ACE-r-group-360', 'ACE-r-group-361', 'ACE-r-group-362', 'ACE-r-group-363', 'ACE-r-group-364',
        'ACE-r-group-365', 'ACE-r-group-366', 'ACE-r-group-367', 'ACE-r-group-368', 'ACE-r-group-369',
        'ACE-r-group-370', 'ACE-r-group-371', 'ACE-r-group-372', 'ACE-r-group-373', 'ACE-r-group-374',
        'ACE-r-group-375', 'ACE-r-group-376', 'ACE-r-group-377', 'ACE-r-group-378', 'ACE-r-group-379',
        'ACE-r-group-380', 'ACE-r-group-381', 'ACE-r-group-382', 'ACE-r-group-383', 'ACE-r-group-384',
        'ACE-r-group-385', 'ACE-r-group-386', 'ACE-r-group-387', 'ACE-r-group-388', 'ACE-r-group-389',
        'ACE-r-group-390', 'ACE-r-group-391', 'ACE-r-group-392', 'ACE-r-group-393', 'ACE-r-group-394',
        'ACE-r-group-395', 'ACE-r-group-396', 'ACE-r-group-397', 'ACE-r-group-398', 'ACE-r-group-399',
        'ACE-r-group-400', 'ACE-r-group-401', 'ACE-r-group-402', 'ACE-r-group-403', 'ACE-r-group-404',
        'ACE-r-group-405', 'ACE-r-group-406', 'ACE-r-group-407', 'ACE-r-group-408', 'ACE-r-group-409',
        'ACE-r-group-410', 'ACE-r-group-411', 'ACE-r-group-412', 'ACE-r-group-413', 'ACE-r-group-414',
        'ACE-r-group-415', 'ACE-r-group-416', 'ACE-r-group-417', 'ACE-r-group-418', 'ACE-r-group-419',
        'ACE-r-group-420', 'ACE-r-group-421', 'ACE-r-group-422', 'ACE-r-group-423', 'ACE-r-group-424',
        'ACE-r-group-425', 'ACE-r-group-426', 'ACE-r-group-427', 'ACE-r-group-428', 'ACE-r-group-429',
        'ACE-r-group-430', 'ACE-r-group-431', 'ACE-r-group-432', 'ACE-r-group-433', 'ACE-r-group-434',
        'ACE-r-group-435', 'ACE-r-group-436', 'ACE-r-group-437', 'ACE-r-group-438', 'ACE-r-group-439',
        'ACE-r-group-440', 'ACE-r-group-441', 'ACE-r-group-442', 'ACE-r-group-443', 'ACE-r-group-444',
        'ACE-r-group-445', 'ACE-r-group-446', 'ACE-r-group-447', 'ACE-r-group-448', 'ACE-r-group-449',
        'ACE-r-group-450', 'ACE-r-group-451', 'ACE-r-group-452', 'ACE-r-group-453', 'ACE-r-group-454',
        'ACE-r-group-455', 'ACE-r-group-456', 'ACE-r-group-457', 'ACE-r-group-458', 'ACE-r-group-459',
        'ACE-r-group-460', 'ACE-r-group-461', 'ACE-r-group-462', 'ACE-r-group-463', 'ACE-r-group-464',
        'ACE-r-group-465', 'ACE-r-group-466', 'ACE-r-group-467', 'ACE-r-group-468', 'ACE-r-group-469',
        'ACE-r-group-470', 'ACE-r-group-471', 'ACE-r-group-472', 'ACE-r-group-473', 'ACE-r-group-474',

        'SIJ-l-group-1', 'SIJ-l-group-2', 'SIJ-l-group-3', 'SIJ-l-group-4', 'SIJ-l-group-5', 'SIJ-l-group-6',
        'SIJ-l-group-7', 'SIJ-l-group-8', 'SIJ-l-group-9', 'SIJ-l-group-10', 'SIJ-l-group-11', 'SIJ-l-group-12',
        'SIJ-l-group-13', 'SIJ-l-group-14', 'SIJ-l-group-15', 'SIJ-l-group-16', 'SIJ-l-group-17', 'SIJ-l-group-18',
        'SIJ-l-group-19', 'SIJ-l-group-20', 'SIJ-l-group-21', 'SIJ-l-group-22', 'SIJ-l-group-23', 'SIJ-l-group-24',
        'SIJ-l-group-25', 'SIJ-l-group-26', 'SIJ-l-group-27', 'SIJ-l-group-28', 'SIJ-l-group-29', 'SIJ-l-group-30',
        'SIJ-l-group-31', 'SIJ-l-group-32', 'SIJ-l-group-33', 'SIJ-l-group-34', 'SIJ-l-group-35', 'SIJ-l-group-36',
        'SIJ-l-group-37', 'SIJ-l-group-38', 'SIJ-l-group-39', 'SIJ-l-group-40', 'SIJ-l-group-41', 'SIJ-l-group-42',
        'SIJ-l-group-43', 'SIJ-l-group-44', 'SIJ-l-group-45', 'SIJ-l-group-46', 'SIJ-l-group-47', 'SIJ-l-group-48',
        'SIJ-l-group-49', 'SIJ-l-group-50', 'SIJ-l-group-51', 'SIJ-l-group-52', 'SIJ-l-group-53', 'SIJ-l-group-54',
        'SIJ-l-group-55', 'SIJ-l-group-56', 'SIJ-l-group-57', 'SIJ-l-group-58', 'SIJ-l-group-59', 'SIJ-l-group-60',
        'SIJ-l-group-61', 'SIJ-l-group-62', 'SIJ-l-group-63', 'SIJ-l-group-64', 'SIJ-l-group-65', 'SIJ-l-group-66',
        'SIJ-l-group-67', 'SIJ-l-group-68', 'SIJ-l-group-69', 'SIJ-l-group-70', 'SIJ-l-group-71', 'SIJ-l-group-72',
        'SIJ-l-group-73', 'SIJ-l-group-74', 'SIJ-l-group-75', 'SIJ-l-group-76', 'SIJ-l-group-77', 'SIJ-l-group-78',
        'SIJ-l-group-79', 'SIJ-l-group-80', 'SIJ-l-group-81', 'SIJ-l-group-82', 'SIJ-l-group-83', 'SIJ-l-group-84',
        'SIJ-l-group-85', 'SIJ-l-group-86', 'SIJ-l-group-87', 'SIJ-l-group-88', 'SIJ-l-group-89', 'SIJ-l-group-90',
        'SIJ-l-group-91', 'SIJ-l-group-92', 'SIJ-l-group-93', 'SIJ-l-group-94', 'SIJ-l-group-95', 'SIJ-l-group-96',
        'SIJ-l-group-97', 'SIJ-l-group-98', 'SIJ-l-group-99', 'SIJ-l-group-100', 'SIJ-l-group-101', 'SIJ-l-group-102',
        'SIJ-l-group-103', 'SIJ-l-group-104', 'SIJ-l-group-105', 'SIJ-l-group-106', 'SIJ-l-group-107', 'SIJ-l-group-108',
        'SIJ-l-group-109', 'SIJ-l-group-110', 'SIJ-l-group-111', 'SIJ-l-group-112', 'SIJ-l-group-113', 'SIJ-l-group-114',
        'SIJ-l-group-115', 'SIJ-l-group-116', 'SIJ-l-group-117', 'SIJ-l-group-118', 'SIJ-l-group-119', 'SIJ-l-group-120',
        'SIJ-l-group-121', 'SIJ-l-group-122', 'SIJ-l-group-123', 'SIJ-l-group-124', 'SIJ-l-group-125', 'SIJ-l-group-126',
        'SIJ-l-group-127', 'SIJ-l-group-128', 'SIJ-l-group-129', 'SIJ-l-group-130', 'SIJ-l-group-131', 'SIJ-l-group-132',
        'SIJ-l-group-133', 'SIJ-l-group-134', 'SIJ-l-group-135', 'SIJ-l-group-136', 'SIJ-l-group-137', 'SIJ-l-group-138',
        'SIJ-l-group-139', 'SIJ-l-group-140', 'SIJ-l-group-141', 'SIJ-l-group-142', 'SIJ-l-group-143', 'SIJ-l-group-144',
        'SIJ-l-group-145', 'SIJ-l-group-146', 'SIJ-l-group-147', 'SIJ-l-group-148', 'SIJ-l-group-149', 'SIJ-l-group-150',
        'SIJ-l-group-151', 'SIJ-l-group-152', 'SIJ-l-group-153', 'SIJ-l-group-154', 'SIJ-l-group-155', 'SIJ-l-group-156',
        'SIJ-l-group-157', 'SIJ-l-group-158', 'SIJ-l-group-159', 'SIJ-l-group-160', 'SIJ-l-group-161', 'SIJ-l-group-162',
        'SIJ-l-group-163', 'SIJ-l-group-164', 'SIJ-l-group-165', 'SIJ-l-group-166', 'SIJ-l-group-167', 'SIJ-l-group-168',
        'SIJ-l-group-169', 'SIJ-l-group-170', 'SIJ-l-group-171', 'SIJ-l-group-172', 'SIJ-l-group-173', 'SIJ-l-group-174',
        'SIJ-l-group-175', 'SIJ-l-group-176', 'SIJ-l-group-177', 'SIJ-l-group-178', 'SIJ-l-group-179', 'SIJ-l-group-180',
        'SIJ-l-group-181', 'SIJ-l-group-182', 'SIJ-l-group-183', 'SIJ-l-group-184', 'SIJ-l-group-185', 'SIJ-l-group-186',
        'SIJ-l-group-187', 'SIJ-l-group-188', 'SIJ-l-group-189', 'SIJ-l-group-190', 'SIJ-l-group-191', 'SIJ-l-group-192',
        'SIJ-l-group-193', 'SIJ-l-group-194', 'SIJ-l-group-195', 'SIJ-l-group-196', 'SIJ-l-group-197', 'SIJ-l-group-198',
        'SIJ-l-group-199', 'SIJ-l-group-200', 'SIJ-l-group-201', 'SIJ-l-group-202', 'SIJ-l-group-203', 'SIJ-l-group-204',
        'SIJ-l-group-205', 'SIJ-l-group-206', 'SIJ-l-group-207', 'SIJ-l-group-208', 'SIJ-l-group-209', 'SIJ-l-group-210',
        'SIJ-l-group-211', 'SIJ-l-group-212', 'SIJ-l-group-213', 'SIJ-l-group-214', 'SIJ-l-group-215', 'SIJ-l-group-216',
        'SIJ-l-group-217', 'SIJ-l-group-218', 'SIJ-l-group-219', 'SIJ-l-group-220', 'SIJ-l-group-221', 'SIJ-l-group-222',
        'SIJ-l-group-223', 'SIJ-l-group-224', 'SIJ-l-group-225', 'SIJ-l-group-226', 'SIJ-l-group-227', 'SIJ-l-group-228',
        'SIJ-l-group-229', 'SIJ-l-group-230', 'SIJ-l-group-231', 'SIJ-l-group-232', 'SIJ-l-group-233', 'SIJ-l-group-234',
        'SIJ-l-group-235', 'SIJ-l-group-236', 'SIJ-l-group-237', 'SIJ-l-group-238', 'SIJ-l-group-239', 'SIJ-l-group-240',
        'SIJ-l-group-241', 'SIJ-l-group-242', 'SIJ-l-group-243', 'SIJ-l-group-244', 'SIJ-l-group-245', 'SIJ-l-group-246',
        'SIJ-l-group-247', 'SIJ-l-group-248', 'SIJ-l-group-249', 'SIJ-l-group-250', 'SIJ-l-group-251', 'SIJ-l-group-252',
        'SIJ-l-group-253', 'SIJ-l-group-254', 'SIJ-l-group-255', 'SIJ-l-group-256', 'SIJ-l-group-257', 'SIJ-l-group-258',
        'SIJ-l-group-259', 'SIJ-l-group-260', 'SIJ-l-group-261', 'SIJ-l-group-262', 'SIJ-l-group-263', 'SIJ-l-group-264',
        'SIJ-l-group-265', 'SIJ-l-group-266', 'SIJ-l-group-267', 'SIJ-l-group-268', 'SIJ-l-group-269', 'SIJ-l-group-270',
        'SIJ-l-group-271', 'SIJ-l-group-272', 'SIJ-l-group-273', 'SIJ-l-group-274', 'SIJ-l-group-275', 'SIJ-l-group-276',
        'SIJ-l-group-277', 'SIJ-l-group-278', 'SIJ-l-group-279', 'SIJ-l-group-280', 'SIJ-l-group-281', 'SIJ-l-group-282',
        'SIJ-l-group-283', 'SIJ-l-group-284', 'SIJ-l-group-285', 'SIJ-l-group-286', 'SIJ-l-group-287', 'SIJ-l-group-288',
        'SIJ-l-group-289', 'SIJ-l-group-290', 'SIJ-l-group-291', 'SIJ-l-group-292',

        'SIJ-r-group-1', 'SIJ-r-group-2', 'SIJ-r-group-3', 'SIJ-r-group-4', 'SIJ-r-group-5', 'SIJ-r-group-6',
        'SIJ-r-group-7', 'SIJ-r-group-8', 'SIJ-r-group-9', 'SIJ-r-group-10', 'SIJ-r-group-11', 'SIJ-r-group-12',
        'SIJ-r-group-13', 'SIJ-r-group-14', 'SIJ-r-group-15', 'SIJ-r-group-16', 'SIJ-r-group-17', 'SIJ-r-group-18',
        'SIJ-r-group-19', 'SIJ-r-group-20', 'SIJ-r-group-21', 'SIJ-r-group-22', 'SIJ-r-group-23', 'SIJ-r-group-24',
        'SIJ-r-group-25', 'SIJ-r-group-26', 'SIJ-r-group-27', 'SIJ-r-group-28', 'SIJ-r-group-29', 'SIJ-r-group-30',
        'SIJ-r-group-31', 'SIJ-r-group-32', 'SIJ-r-group-33', 'SIJ-r-group-34', 'SIJ-r-group-35', 'SIJ-r-group-36',
        'SIJ-r-group-37', 'SIJ-r-group-38', 'SIJ-r-group-39', 'SIJ-r-group-40', 'SIJ-r-group-41', 'SIJ-r-group-42',
        'SIJ-r-group-43', 'SIJ-r-group-44', 'SIJ-r-group-45', 'SIJ-r-group-46', 'SIJ-r-group-47', 'SIJ-r-group-48',
        'SIJ-r-group-49', 'SIJ-r-group-50', 'SIJ-r-group-51', 'SIJ-r-group-52', 'SIJ-r-group-53', 'SIJ-r-group-54',
        'SIJ-r-group-55', 'SIJ-r-group-56', 'SIJ-r-group-57', 'SIJ-r-group-58', 'SIJ-r-group-59', 'SIJ-r-group-60',
        'SIJ-r-group-61', 'SIJ-r-group-62', 'SIJ-r-group-63', 'SIJ-r-group-64', 'SIJ-r-group-65', 'SIJ-r-group-66',
        'SIJ-r-group-67', 'SIJ-r-group-68', 'SIJ-r-group-69', 'SIJ-r-group-70', 'SIJ-r-group-71', 'SIJ-r-group-72',
        'SIJ-r-group-73', 'SIJ-r-group-74', 'SIJ-r-group-75', 'SIJ-r-group-76', 'SIJ-r-group-77', 'SIJ-r-group-78',
        'SIJ-r-group-79', 'SIJ-r-group-80', 'SIJ-r-group-81', 'SIJ-r-group-82', 'SIJ-r-group-83', 'SIJ-r-group-84',
        'SIJ-r-group-85', 'SIJ-r-group-86', 'SIJ-r-group-87', 'SIJ-r-group-88', 'SIJ-r-group-89', 'SIJ-r-group-90',
        'SIJ-r-group-91', 'SIJ-r-group-92', 'SIJ-r-group-93', 'SIJ-r-group-94', 'SIJ-r-group-95', 'SIJ-r-group-96',
        'SIJ-r-group-97', 'SIJ-r-group-98', 'SIJ-r-group-99', 'SIJ-r-group-100', 'SIJ-r-group-101', 'SIJ-r-group-102',
        'SIJ-r-group-103', 'SIJ-r-group-104', 'SIJ-r-group-105', 'SIJ-r-group-106', 'SIJ-r-group-107',
        'SIJ-r-group-108', 'SIJ-r-group-109', 'SIJ-r-group-110', 'SIJ-r-group-111', 'SIJ-r-group-112',
        'SIJ-r-group-113', 'SIJ-r-group-114', 'SIJ-r-group-115', 'SIJ-r-group-116', 'SIJ-r-group-117',
        'SIJ-r-group-118', 'SIJ-r-group-119', 'SIJ-r-group-120', 'SIJ-r-group-121', 'SIJ-r-group-122',
        'SIJ-r-group-123', 'SIJ-r-group-124', 'SIJ-r-group-125', 'SIJ-r-group-126', 'SIJ-r-group-127',
        'SIJ-r-group-128', 'SIJ-r-group-129', 'SIJ-r-group-130', 'SIJ-r-group-131', 'SIJ-r-group-132',
        'SIJ-r-group-133', 'SIJ-r-group-134', 'SIJ-r-group-135', 'SIJ-r-group-136', 'SIJ-r-group-137',
        'SIJ-r-group-138', 'SIJ-r-group-139', 'SIJ-r-group-140', 'SIJ-r-group-141', 'SIJ-r-group-142',
        'SIJ-r-group-143', 'SIJ-r-group-144', 'SIJ-r-group-145', 'SIJ-r-group-146', 'SIJ-r-group-147',
        'SIJ-r-group-148', 'SIJ-r-group-149', 'SIJ-r-group-150', 'SIJ-r-group-151', 'SIJ-r-group-152',
        'SIJ-r-group-153', 'SIJ-r-group-154', 'SIJ-r-group-155', 'SIJ-r-group-156', 'SIJ-r-group-157',
        'SIJ-r-group-158', 'SIJ-r-group-159', 'SIJ-r-group-160', 'SIJ-r-group-161', 'SIJ-r-group-162',
        'SIJ-r-group-163', 'SIJ-r-group-164', 'SIJ-r-group-165', 'SIJ-r-group-166', 'SIJ-r-group-167',
        'SIJ-r-group-168', 'SIJ-r-group-169', 'SIJ-r-group-170', 'SIJ-r-group-171', 'SIJ-r-group-172',
        'SIJ-r-group-173', 'SIJ-r-group-174', 'SIJ-r-group-175', 'SIJ-r-group-176', 'SIJ-r-group-177',
        'SIJ-r-group-178', 'SIJ-r-group-179', 'SIJ-r-group-180', 'SIJ-r-group-181', 'SIJ-r-group-182',
        'SIJ-r-group-183', 'SIJ-r-group-184', 'SIJ-r-group-185', 'SIJ-r-group-186', 'SIJ-r-group-187',
        'SIJ-r-group-188', 'SIJ-r-group-189', 'SIJ-r-group-190', 'SIJ-r-group-191', 'SIJ-r-group-192',
        'SIJ-r-group-193', 'SIJ-r-group-194', 'SIJ-r-group-195', 'SIJ-r-group-196', 'SIJ-r-group-197',
        'SIJ-r-group-198', 'SIJ-r-group-199', 'SIJ-r-group-200', 'SIJ-r-group-201', 'SIJ-r-group-202',
        'SIJ-r-group-203', 'SIJ-r-group-204', 'SIJ-r-group-205', 'SIJ-r-group-206', 'SIJ-r-group-207',
        'SIJ-r-group-208', 'SIJ-r-group-209', 'SIJ-r-group-210', 'SIJ-r-group-211', 'SIJ-r-group-212',
        'SIJ-r-group-213', 'SIJ-r-group-214', 'SIJ-r-group-215', 'SIJ-r-group-216', 'SIJ-r-group-217',
        'SIJ-r-group-218', 'SIJ-r-group-219', 'SIJ-r-group-220', 'SIJ-r-group-221', 'SIJ-r-group-222',
        'SIJ-r-group-223', 'SIJ-r-group-224', 'SIJ-r-group-225', 'SIJ-r-group-226', 'SIJ-r-group-227',
        'SIJ-r-group-228', 'SIJ-r-group-229', 'SIJ-r-group-230', 'SIJ-r-group-231', 'SIJ-r-group-232',
        'SIJ-r-group-233', 'SIJ-r-group-234', 'SIJ-r-group-235', 'SIJ-r-group-236', 'SIJ-r-group-237',
        'SIJ-r-group-238', 'SIJ-r-group-239', 'SIJ-r-group-240', 'SIJ-r-group-241', 'SIJ-r-group-242',
        'SIJ-r-group-243', 'SIJ-r-group-244', 'SIJ-r-group-245', 'SIJ-r-group-246', 'SIJ-r-group-247',
        'SIJ-r-group-248', 'SIJ-r-group-249', 'SIJ-r-group-250', 'SIJ-r-group-251', 'SIJ-r-group-252',
        'SIJ-r-group-253', 'SIJ-r-group-254', 'SIJ-r-group-255', 'SIJ-r-group-256', 'SIJ-r-group-257',
        'SIJ-r-group-258', 'SIJ-r-group-259', 'SIJ-r-group-260', 'SIJ-r-group-261', 'SIJ-r-group-262',
        'SIJ-r-group-263', 'SIJ-r-group-264', 'SIJ-r-group-265', 'SIJ-r-group-266', 'SIJ-r-group-267',
        'SIJ-r-group-268', 'SIJ-r-group-269', 'SIJ-r-group-270', 'SIJ-r-group-271', 'SIJ-r-group-272',
        'SIJ-r-group-273', 'SIJ-r-group-274', 'SIJ-r-group-275', 'SIJ-r-group-276',

        'PS-l-group-1', 'PS-l-group-2', 'PS-l-group-3', 'PS-l-group-4', 'PS-l-group-5', 'PS-l-group-6', 'PS-l-group-7',
        'PS-l-group-8', 'PS-l-group-9', 'PS-l-group-10', 'PS-l-group-11', 'PS-l-group-12', 'PS-l-group-13',
        'PS-l-group-14', 'PS-l-group-15', 'PS-l-group-16', 'PS-l-group-17', 'PS-l-group-18', 'PS-l-group-19',
        'PS-l-group-20', 'PS-l-group-21', 'PS-l-group-22', 'PS-l-group-23', 'PS-l-group-24', 'PS-l-group-25',
        'PS-l-group-26', 'PS-l-group-27', 'PS-l-group-28', 'PS-l-group-29', 'PS-l-group-30', 'PS-l-group-31',
        'PS-l-group-32', 'PS-l-group-33', 'PS-l-group-34', 'PS-l-group-35', 'PS-l-group-36', 'PS-l-group-37',
        'PS-l-group-38', 'PS-l-group-39', 'PS-l-group-40', 'PS-l-group-41', 'PS-l-group-42', 'PS-l-group-43',
        'PS-l-group-44', 'PS-l-group-45', 'PS-l-group-46', 'PS-l-group-47', 'PS-l-group-48', 'PS-l-group-49',
        'PS-l-group-50', 'PS-l-group-51', 'PS-l-group-52', 'PS-l-group-53', 'PS-l-group-54', 'PS-l-group-55',
        'PS-l-group-56', 'PS-l-group-57', 'PS-l-group-58', 'PS-l-group-59', 'PS-l-group-60', 'PS-l-group-61',
        'PS-l-group-62', 'PS-l-group-63', 'PS-l-group-64', 'PS-l-group-65', 'PS-l-group-66', 'PS-l-group-67',
        'PS-l-group-68', 'PS-l-group-69', 'PS-l-group-70', 'PS-l-group-71', 'PS-l-group-72', 'PS-l-group-73',
        'PS-l-group-74', 'PS-l-group-75', 'PS-l-group-76', 'PS-l-group-77', 'PS-l-group-78', 'PS-l-group-79',
        'PS-l-group-80', 'PS-l-group-81', 'PS-l-group-82', 'PS-l-group-83', 'PS-l-group-84', 'PS-l-group-85',
        'PS-l-group-86', 'PS-l-group-87', 'PS-l-group-88', 'PS-l-group-89', 'PS-l-group-90', 'PS-l-group-91',
        'PS-l-group-92', 'PS-l-group-93', 'PS-l-group-94', 'PS-l-group-95', 'PS-l-group-96', 'PS-l-group-97',
        'PS-l-group-98', 'PS-l-group-99',

        'glut_med1_r-P1', 'glut_med2_r-P1', 'glut_med3_r-P1', 'glut_min1_r-P1', 'glut_min2_r-P1',
        'glut_min3_r-P1', 'semimem_r-P1', 'semiten_r-P1', 'bifemlh_r-P1', 'sar_r-P1',
        'ADD_LONG_r-P1', 'add_brev_r-P1', 'add_mag1_r-P1', 'add_mag2_r-P1', 'add_mag3_r-P1',
        'tfl_r-P1', 'pect_r-P1', 'grac_r-P1', 'glut_max1_r-P1', 'glut_max2_r-P1',
        'glut_max3_r-P1', 'iliacus_r-P1', 'quad_fem_r-P1', 'gem_r-P1', 'peri_r-P1',
        'peri_r-P2', 'rect_fem_r-P1', 'rect_abd_r-P1', 'QL_post_I_1-L3_r-P1', 'QL_post_I_2-L4_r-P1',
        'QL_post_I_2-L3_r-P1', 'QL_post_I_2-L2_r-P1', 'QL_post_I_3-L1_r-P1',
        'QL_post_I_3-L2_r-P1', 'QL_post_I_3-L3_r-P1', 'QL_ant_I_2-T12_r-P1', 'QL_ant_I_3-T12_r-P1',
        'QL_ant_I_2-12_1_r-P2', 'QL_ant_I_3-12_1_r-P1', 'QL_ant_I_3-12_2_r-P1',
        'QL_ant_I_3-12_3_r-P2', 'MF_m1t_2_r-P1', 'MF_m1t_3_r-P1', 'MF_m2t_1_r-P1',
        'MF_m2t_2_r-P1', 'MF_m2t_3_r-P1', 'MF_m3s_r-P1', 'MF_m3t_1_r-P1', 'MF_m3t_2_r-P1',
        'MF_m3t_3_r-P1', 'MF_m4s_r-P1', 'MF_m4t_1_r-P1', 'MF_m4t_2_r-P1', 'MF_m4t_3_r-P1',
        'MF_m5s_r-P1', 'MF_m5t_1_r-P1', 'MF_m5t_2_r-P1', 'MF_m5t_3_r-P1', 'MF_M4_LAMINAR_r-P1',
        'MF_M5_LAMINAR_r-P1', 'IL_L1_r-P1', 'IL_L2_r-P1', 'IL_L3_r-P1', 'IL_L4_r-P1',
        'IL_R5_r-P1', 'IL_R6_r-P1', 'IL_R7_r-P1', 'IL_R8_r-P1',
        'IL_R9_r-P1', 'IL_R10_r-P1', 'IL_R11_r-P1', 'IL_R12_r-P1',
        'LTpT_T7_r-P1', 'LTpT_T8_r-P1', 'LTpT_T9_r-P1',
        'LTpT_T10_r-P1', 'LTpT_T11_r-P1', 'LTpT_T12_r-P1',
        'LTpT_R7_r-P1', 'LTpT_R8_r-P1', 'LTpT_R9_r-P1',
        'LTpT_R10_r-P1', 'LTpT_R11_r-P1', 'LTpT_R12_r-P1',
        'LTpL_L5_r-P1', 'LTpL_L4_r-P1', 'LTpL_L3_r-P1', 'LTpL_L2_r-P1', 'LTpL_L1_r-P1',
        'EO5_r-P1', 'EO6_r-P1',
        'IO1_r-P1', 'IO2_r-P1', 'IO3_r-P1', 'IO4_r-P1', 'IO5_r-P1', 'IO6_r-P1', 'LD_Il_r-P1'
    ]

    landmark_displacement_outpath = rf"J:\PG_Pelvis\WholePelvis\PCA_predict\output\landmark_nodes_displacement_{prefix}.txt"
    with open(landmark_displacement_outpath, 'w') as f:
        f.write("# landmark_name landmark_node_index x y z dx dy dz\n")
        for i, (idx, new_coord, disp) in enumerate(zip(vertex_indices,
                                                       landmark_nodes_new_coords,
                                                       landmark_nodes_displacement)):
            name = landmark_names[i] if i < len(landmark_names) else f"landmark-{i}"
            f.write(f"{name} {idx} {new_coord[0]:.3f} {new_coord[1]:.3f} {new_coord[2]:.3f} "
                    f"{disp[0]:.3f} {disp[1]:.3f} {disp[2]:.3f}\n")
    print(f"[INFO] Landmark displacement saved: {landmark_displacement_outpath}")

    name_to_index = {name: i for i, name in enumerate(landmark_names)}

    def get_point(name):
        if name not in name_to_index:
            raise ValueError(f"lack of landmark: {name}")
        idx = name_to_index[name]
        return landmark_nodes_new_coords[idx]

    P_RASIS = get_point('pelvis-RASIS')
    P_LASIS = get_point('pelvis-LASIS')

    pubic_names = ['pubic-L1', 'pubic-R1']
    pubic_pts = np.array([get_point(n) for n in pubic_names])
    P_pubic_mid = pubic_pts.mean(axis=0)

    v_a = P_pubic_mid - P_RASIS
    v_b = P_pubic_mid - P_LASIS

    def compute_angle(v):
        y, z = abs(v[1]), abs(v[2])
        angle_rad = np.arctan2(z, y)
        return np.degrees(angle_rad)

    angle_a = compute_angle(v_a)
    angle_b = compute_angle(v_b)

    angle_mean = (angle_a + angle_b) / 2.0

    print(f"[INFO] Plane A angle (RASIS): {angle_a:.3f} deg")
    print(f"[INFO] Plane B angle (LASIS): {angle_b:.3f} deg")
    print(f"[INFO] Mean angle: {angle_mean:.3f} deg")

    angle_outpath = rf"J:\PG_Pelvis\WholePelvis\PCA_predict\output\pelvis_angles_{prefix}.txt"

    with open(angle_outpath, 'w') as f:
        f.write("# prefix plane angle_deg\n")
        f.write(f"{prefix} plane_A_RASIS {angle_a:.6f}\n")
        f.write(f"{prefix} plane_B_LASIS {angle_b:.6f}\n")
        f.write(f"{prefix} plane_mean {angle_mean:.6f}\n")

    print(f"[INFO] Angle file saved: {angle_outpath}")


if __name__ == "__main__":
    init_log(None)

    files = sorted(glob.glob(os.path.join(landmark_targets_dir, "p*.txt")))

    print(f"Detected {len(files)} files.")

    for f in files:
        try:
            landmark_targets = f  
            print("\nRunning:", os.path.basename(f))
            run_registration()
        except Exception as e:
            print("Failed:", os.path.basename(f), e)
