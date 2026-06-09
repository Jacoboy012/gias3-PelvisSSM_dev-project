%% ================= CONFIGURATION =================
txt_folder = 'J:\PG_Pelvis\WholePelvis\PCA_predict\output';
txt_pattern = '*_p*.txt';
osim_template = 'J:\PG_Pelvis\OpensimModel\PG-model.osim';
output_folder = 'J:\PG_Pelvis\WholePelvis\PCA_predict\output\split_pelvis\import_pelvis_into_model';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

txt_files = dir(fullfile(txt_folder, txt_pattern));

%% ================= PROCESS EACH TXT FILE =================
for i = 1:length(txt_files)
    txt_name = txt_files(i).name;
    txt_path = fullfile(txt_folder, txt_name);

    %% ================= READ LANDMARK TXT AND COMPUTE KEY POINTS =================
    fid = fopen(txt_path,'r');
    landmark_dict = containers.Map;
    lines = {};  % Store all lines for later processing
    tline = fgetl(fid);
    while ischar(tline)
        lines{end+1} = tline; % Store original line

        tline_trim = strtrim(tline);
        if isempty(tline_trim) || startsWith(tline_trim,'#')
            tline = fgetl(fid);
            continue
        end
        parts = strsplit(tline_trim);
        name = parts{1};
        coords = str2double(parts(3:5))/1000; % Convert mm to meters
        landmark_dict(name) = coords;

        tline = fgetl(fid);
    end
    fclose(fid);

    fprintf('Landmarks in file %s:\n', txt_name);
    disp(keys(landmark_dict));

    %%% ------------------ COMPUTE KEY LANDMARKS ------------------ %%%

    all_keys = keys(landmark_dict);

    % SIJ joint centers (Sacroiliac Joint)
    sacrumjnt_l = 0.5*(landmark_dict('SIJ-L1') + landmark_dict('SIJ-L2'));
    sacrumjnt_r = 0.5*(landmark_dict('SIJ-R1') + landmark_dict('SIJ-R2'));

    % Pubic symphysis mean point
    pubic_pts = [landmark_dict('pubic-L1'); landmark_dict('pubic-R1'); ];
    pubic_mean = mean(pubic_pts,1);

    % L5/S1 joint center and plane fitting
    L5_S1_keys = all_keys(contains(all_keys, 'L5-S1-group'));
    if isempty(L5_S1_keys)
        warning('No L5-S1-group points found, using single point L5-S1 as fallback');
        L5_S1_IVDjnt = landmark_dict('L5-S1');
        n_l5s1 = [0; 1; 0];  % Default vertical upward direction
    else
        L5_S1_pts = cell2mat(values(landmark_dict, L5_S1_keys)');
        L5_S1_IVDjnt = mean(L5_S1_pts, 1);
        [n_l5s1, ~] = fitPlaneNormal(L5_S1_pts);
        fprintf('L5/S1 joint center computed from %d points: [%.6f %.6f %.6f]\n', ...
            length(L5_S1_keys), L5_S1_IVDjnt);
        fprintf('L5/S1 plane normal vector (Y-axis): [%.6f %.6f %.6f]\n', n_l5s1);
    end

    %% ================= CONSTRUCT ANTERIOR PELVIS REFERENCE PLANE (for AnatFrame) =================

    % Extract PS-l-group points (Pubic Symphysis left group)
    PS_l_keys = all_keys(contains(all_keys, 'PS-l-group'));
    if isempty(PS_l_keys)
        error('PS-l-group points not found. Please check the input file.');
    end
    PS_l_pts = cell2mat(values(landmark_dict, PS_l_keys)');

    % Find the most anterior point (maximum x-coordinate)
    [~, idx_front] = max(PS_l_pts(:,1));
    PS_front = PS_l_pts(idx_front,:);

    % Extract ASIS landmarks (Anterior Superior Iliac Spine)
    ASIS_L = landmark_dict('pelvis-LASIS');
    ASIS_R = landmark_dict('pelvis-RASIS');

    % Define the reference plane
    v1 = ASIS_R - ASIS_L;
    v2 = PS_front - ASIS_L;
    plane_normal = cross(v1, v2);
    plane_normal = plane_normal / norm(plane_normal);
    plane_origin = 0.5 * (ASIS_L + ASIS_R);

    %%% ------------------ HIP JOINT CENTER SPHERE FITTING (with radius constraints) ------------------ %%%
    fitSphereWithRadiusBounds = @(P, r_bounds) deal(lsqSphere(P, r_bounds));

    landmarks = struct();
    landmarks.lengths = struct();

    ACE_l_keys = all_keys(contains(all_keys, 'ACE-l-group'));
    ACE_r_keys = all_keys(contains(all_keys, 'ACE-r-group'));

    if isempty(ACE_l_keys) || isempty(ACE_r_keys)
        error('ACE-l-group or ACE-r-group points not found. Please check the input file.');
    end

    ACE_l_pts = cell2mat(values(landmark_dict, ACE_l_keys)');
    ACE_r_pts = cell2mat(values(landmark_dict, ACE_r_keys)');

    r_bounds = [0.015, 0.045];  % Anatomically plausible hip joint radius range (15-45 mm)

    [centre_l, rad_l] = fitSphereWithRadiusBounds(ACE_l_pts, r_bounds);
    landmarks.LHJC.coords = centre_l';
    landmarks.LHJC.radius = rad_l;
    landmarks.lengths.LHJ_diameter = 2 * rad_l;

    [centre_r, rad_r] = fitSphereWithRadiusBounds(ACE_r_pts, r_bounds);
    landmarks.RHJC.coords = centre_r';
    landmarks.RHJC.radius = rad_r;
    landmarks.lengths.RHJ_diameter = 2 * rad_r;

    hip_l = landmarks.LHJC.coords;
    hip_r = landmarks.RHJC.coords;

    fprintf('Left hip joint center: [%.6f %.6f %.6f], radius = %.4f m\n', hip_l, rad_l);
    fprintf('Right hip joint center: [%.6f %.6f %.6f], radius = %.4f m\n', hip_r, rad_r);

    % Extract mesh filename prefix
    m = regexp(txt_name, '_p\d{3}', 'match');
    if ~isempty(m)
        prefix = m{1}(2:end);
    else
        error('Cannot extract pXX prefix from %s', txt_name);
    end
    sacrum_mesh = sprintf('%s_sac.obj', prefix);
    pelvis_l_mesh = sprintf('%s_pel_l.obj', prefix);
    pelvis_r_mesh = sprintf('%s_pel_r.obj', prefix);

    %% ================= LOAD BASE OSIM XML TEMPLATE =================
    xDoc_base = xmlread(osim_template);

    %% ================= EXTRACT COMMON DATA: SIJ PLANES & PUBIC LONG AXIS =================
    jointSet = xDoc_base.getElementsByTagName('JointSet').item(0);

    SIJ_l_keys = all_keys(contains(all_keys, 'SIJ-l-group'));
    SIJ_r_keys = all_keys(contains(all_keys, 'SIJ-r-group'));

    if isempty(SIJ_l_keys) || isempty(SIJ_r_keys)
        error('SIJ-l-group or SIJ-r-group points not found. Please check the input file.');
    end

    SIJ_l_pts = cell2mat(values(landmark_dict, SIJ_l_keys)');
    SIJ_r_pts = cell2mat(values(landmark_dict, SIJ_r_keys)');

    [n_l, origin_l] = fitPlaneNormal(SIJ_l_pts);
    [n_r, origin_r] = fitPlaneNormal(SIJ_r_pts);

    % Compute pubic long axis direction
    PS_l_keys = all_keys(contains(all_keys, 'PS-l-group'));
    if isempty(PS_l_keys)
        error('PS-l-group points not found. Please check the input file.');
    end
    PS_l_pts = cell2mat(values(landmark_dict, PS_l_keys)');

    D = pdist2(PS_l_pts, PS_l_pts);
    [~, maxIdx] = max(D(:));
    [i1, i2] = ind2sub(size(D), maxIdx);
    long_axis = PS_l_pts(i2,:) - PS_l_pts(i1,:);
    long_axis = long_axis / norm(long_axis);

    a = long_axis;
    a(3) = 0;  % Project onto XY plane
    if norm(a) < 1e-8
        [~,~,Vps] = svd(bsxfun(@minus, PS_l_pts, mean(PS_l_pts,1)), 0);
        pc1 = Vps(:,1)';
        a = pc1; a(3) = 0;
        if norm(a) < 1e-8
            warning('Pubic long axis projection degenerate, using X-axis as default');
            a = [1 0 0];
        end
    end
    a = a / norm(a);

    % Construct right-handed coordinate system (Z-axis upward)
    z_axis_pubic = [0; 0; 1];
    x_axis_pubic = -a(:);
    y_axis_pubic = cross(z_axis_pubic, x_axis_pubic);
    y_axis_pubic = y_axis_pubic / norm(y_axis_pubic);
    x_axis_pubic = cross(y_axis_pubic, z_axis_pubic);
    x_axis_pubic = x_axis_pubic / norm(x_axis_pubic);

    R_pubic = [x_axis_pubic, y_axis_pubic, z_axis_pubic];
    theta_x = atan2(-R_pubic(2,3), R_pubic(3,3));
    theta_y = asin(R_pubic(1,3));
    theta_z = atan2(-R_pubic(1,2), R_pubic(1,1));
    eul_pubic = [theta_x, theta_y, theta_z];

    fprintf('Pubic long axis direction a = [%.4f %.4f %.4f], orientation = [%.4f %.4f %.4f] rad\n', a, eul_pubic);

%% ================= VERSION 1: AnatFrame (Pelvis Reference Plane Constrained) =================
xDoc_anat = xmlread(osim_template);

% SIJ orientation using plane constraint
ori_l_anat = normal2eulerXYZ_constrained(n_l, plane_normal);
ori_r_anat = normal2eulerXYZ_constrained(n_r, plane_normal);

% Pubic orientation using the same pelvis reference plane
eul_pubic_anat = computePubicOrientationFromPlane(PS_l_pts, plane_normal);
fprintf('AnatFrame Pubic orientation = [%.4f %.4f %.4f] rad\n', eul_pubic_anat);

% Compute L5/S1 orientation (unified method)
ori_l5s1 = normal2eulerXYZ_L5S1_unified(n_l5s1, plane_normal);

modifyJointTransforms(xDoc_anat, sacrumjnt_l, sacrumjnt_r, ori_l_anat, ori_r_anat, ...
    pubic_mean, eul_pubic_anat, hip_l, hip_r, L5_S1_IVDjnt, ori_l5s1);
modifyBodyMeshes(xDoc_anat, sacrum_mesh, pelvis_l_mesh, pelvis_r_mesh);

% Locate muscle path point start line (from glut_med1_r-P1 onward)
start_line = find_start_line_for_muscles(lines, 'glut_med1_r-P1');
modifyPathPoints(xDoc_anat, lines, start_line);

out_name_anat = sprintf('New_%s_AnatFrame.osim', prefix);
out_path_anat = fullfile(output_folder, out_name_anat);
xmlwrite(out_path_anat, xDoc_anat);
fprintf('Saved: %s\n', out_path_anat);

%% ================= VERSION 2: EvaluatingFrame (Hip Joint Direction Based) =================
xDoc_eval = xmlread(osim_template);

% SIJ orientation using hip joint direction
ori_l_eval = normal2eulerXYZ_withHip(n_l, hip_l, origin_l);
ori_r_eval = normal2eulerXYZ_withHip(n_r, hip_r, origin_r);

% Pubic orientation using long axis projected onto XY plane
eul_pubic_eval = computePubicOrientationFromLongAxis(PS_l_pts);
fprintf('EvaluatingFrame Pubic orientation = [%.4f %.4f %.4f] rad\n', eul_pubic_eval);

% Compute L5/S1 orientation (unified method)
ori_l5s1 = normal2eulerXYZ_L5S1_unified(n_l5s1, plane_normal);

modifyJointTransforms(xDoc_eval, sacrumjnt_l, sacrumjnt_r, ori_l_eval, ori_r_eval, ...
    pubic_mean, eul_pubic_eval, hip_l, hip_r, L5_S1_IVDjnt, ori_l5s1);
modifyBodyMeshes(xDoc_eval, sacrum_mesh, pelvis_l_mesh, pelvis_r_mesh);

% Locate muscle path point start line (from glut_med1_r-P1 onward)
start_line = find_start_line_for_muscles(lines, 'glut_med1_r-P1');
modifyPathPoints(xDoc_eval, lines, start_line);

out_name_eval = sprintf('New_%s_EvaluatingFrame.osim', prefix);
out_path_eval = fullfile(output_folder, out_name_eval);
xmlwrite(out_path_eval, xDoc_eval);
fprintf('Saved: %s\n', out_path_eval);

end

%% ================= HELPER FUNCTIONS =================

function modifyJointTransforms(xDoc, sacrumjnt_l, sacrumjnt_r, ori_l, ori_r, ...
    pubic_mean, eul_pubic, hip_l, hip_r, L5_S1_IVDjnt, ori_l5s1)
    % Modify joint transformations including SIJ, pubic, hip, and L5/S1 joints
    
    jointSet = xDoc.getElementsByTagName('JointSet').item(0);
    joints = jointSet.getElementsByTagName('CustomJoint');
    
    % Modify SIJ joints
    for j = 0:joints.getLength-1
        joint = joints.item(j);
        joint_name = char(joint.getAttribute('name'));
        
        if strcmp(joint_name, 'sacrumjnt_l')
            frames = joint.getElementsByTagName('frames').item(0);
            offsets = frames.getElementsByTagName('PhysicalOffsetFrame');
            for k = 0:offsets.getLength-1
                frame = offsets.item(k);
                trans = frame.getElementsByTagName('translation').item(0);
                trans.getFirstChild.setNodeValue(sprintf('%.6f %.6f %.6f', sacrumjnt_l));
                orient = frame.getElementsByTagName('orientation').item(0);
                if ~isempty(orient)
                    orient.getFirstChild.setNodeValue(sprintf('%.6f %.6f %.6f', ori_l));
                end
            end
        elseif strcmp(joint_name, 'sacrumjnt_r')
            frames = joint.getElementsByTagName('frames').item(0);
            offsets = frames.getElementsByTagName('PhysicalOffsetFrame');
            for k = 0:offsets.getLength-1
                frame = offsets.item(k);
                trans = frame.getElementsByTagName('translation').item(0);
                trans.getFirstChild.setNodeValue(sprintf('%.6f %.6f %.6f', sacrumjnt_r));
                orient = frame.getElementsByTagName('orientation').item(0);
                if ~isempty(orient)
                    orient.getFirstChild.setNodeValue(sprintf('%.6f %.6f %.6f', ori_r));
                end
            end
        end
    end
    
    % Modify Pubic joints
    joint_names = {'pubic_l', 'pubic_r'};
    offset_frame_names = {'pelvis_l_offset', 'pelvis_r_offset'};
    for idx = 1:length(joint_names)
        joint_name = joint_names{idx};
        expected_offset = offset_frame_names{idx};
        
        joint = [];
        for j = 0:joints.getLength-1
            jnt = joints.item(j);
            if strcmp(char(jnt.getAttribute('name')), joint_name)
                joint = jnt; break;
            end
        end
        if isempty(joint), continue; end
        
        frames = joint.getElementsByTagName('frames').item(0);
        offsets = frames.getElementsByTagName('PhysicalOffsetFrame');
        for k = 0:offsets.getLength-1
            frame = offsets.item(k);
            if strcmp(char(frame.getAttribute('name')), expected_offset)
                trans = frame.getElementsByTagName('translation').item(0);
                trans.getFirstChild.setNodeValue(sprintf('%.6f %.6f %.6f', pubic_mean));
                orient = frame.getElementsByTagName('orientation').item(0);
                orient_str = sprintf('%.6f %.6f %.6f', eul_pubic);
                if isempty(orient)
                    orient = xDoc.createElement('orientation');
                    orient_text = xDoc.createTextNode(orient_str);
                    orient.appendChild(orient_text);
                    frame.appendChild(orient);
                else
                    orient.getFirstChild.setNodeValue(orient_str);
                end
                break;
            end
        end
    end
    
    % Modify hip and L5/S1 joints
    hip_names = {'hip_l', 'hip_r'};
    hip_coords = {hip_l, hip_r};
    for idx = 1:length(hip_names)
        modifyTranslation(xDoc, hip_names{idx}, hip_coords{idx}, 'pelvis_l_offset', hip_coords{idx});
    end
    
    % Modify L5/S1 joint (both translation and orientation)
    modifyJointWithOrientation(xDoc, 'L5_S1_IVDjnt', L5_S1_IVDjnt, 'sacrum_offset', ori_l5s1);
end

function modifyTranslation(xDoc, joint_name, coord, offset_name, ~)
    % Modify only the translation of a joint's offset frame
    
    jointSet = xDoc.getElementsByTagName('JointSet').item(0);
    joints = jointSet.getElementsByTagName('CustomJoint');
    
    joint = [];
    for j = 0:joints.getLength-1
        jnt = joints.item(j);
        if strcmp(char(jnt.getAttribute('name')), joint_name)
            joint = jnt; break;
        end
    end
    if isempty(joint), return; end
    
    frames = joint.getElementsByTagName('frames').item(0);
    offsets = frames.getElementsByTagName('PhysicalOffsetFrame');
    for k = 0:offsets.getLength-1
        frame = offsets.item(k);
        if strcmp(char(frame.getAttribute('name')), offset_name)
            trans = frame.getElementsByTagName('translation').item(0);
            trans.getFirstChild.setNodeValue(sprintf('%.6f %.6f %.6f', coord));
            break;
        end
    end
end

function modifyBodyMeshes(xDoc, sacrum_mesh, pelvis_l_mesh, pelvis_r_mesh)
    % Update mesh file references for pelvic bodies
    
    bodySet = xDoc.getElementsByTagName('BodySet').item(0);
    bodies = bodySet.getElementsByTagName('Body');
    
    for j = 0:bodies.getLength-1
        body = bodies.item(j);
        body_name = char(body.getAttribute('name'));
        
        if any(strcmp(body_name, {'sacrum', 'pelvis_l', 'pelvis_r'}))
            attached_geometries = body.getElementsByTagName('attached_geometry');
            if attached_geometries.getLength > 0
                geom = attached_geometries.item(0);
                mesh_node = geom.getElementsByTagName('mesh_file').item(0);
                
                if strcmp(body_name, 'sacrum')
                    mesh_node.getFirstChild.setNodeValue(sacrum_mesh);
                elseif strcmp(body_name, 'pelvis_l')
                    mesh_node.getFirstChild.setNodeValue(pelvis_l_mesh);
                elseif strcmp(body_name, 'pelvis_r')
                    mesh_node.getFirstChild.setNodeValue(pelvis_r_mesh);
                end
                
                scale_node = geom.getElementsByTagName('scale_factors').item(0);
                if ~isempty(scale_node) && ~isempty(scale_node.getFirstChild)
                    scale_node.getFirstChild.setNodeValue('1 1 1');
                end
            end
        end
    end
end

function modifyPathPoints(xDoc, lines, start_line)
    % Update muscle path point locations from landmark data
    % Mirrors right-side points to left side when applicable
    
    muscle_lines = lines(start_line:end);
    muscle_table = {};
    
    % Parse muscle point data
    for li = 1:length(muscle_lines)
        parts = strsplit(strtrim(muscle_lines{li}));
        if length(parts) < 5, continue; end
        name_full = parts{1};
        coords = str2double(parts(3:5))/1000;  % Convert mm to meters
        
        % Extract muscle token and point suffix
        idx = find(name_full == '-', 1, 'last');
        if isempty(idx), continue; end
        token = name_full(1:idx-1);
        Px = name_full(idx:end);
        
        muscle_table(end+1,:) = {token, Px, coords(1), coords(2), coords(3)};
        
        % Mirror right-side points to left side (negate X coordinate)
        if contains(token, '_r')
            token_l = strrep(token, '_r', '_l');
            coords_l = [coords(1), coords(2), -coords(3)];
            muscle_table(end+1,:) = {token_l, Px, coords_l(1), coords_l(2), coords_l(3)};
        end
    end
    
    % Create full name mapping
    for mi = 1:size(muscle_table,1)
        muscle_table{mi,6} = [muscle_table{mi,1}, muscle_table{mi,2}];
    end
    
    % Update muscle path points in the OpenSim model
    forceSet = xDoc.getElementsByTagName('ForceSet').item(0);
    muscles_xml = forceSet.getElementsByTagName('Thelen2003Muscle');
    
    for mi = 0:muscles_xml.getLength-1
        muscle = muscles_xml.item(mi);
        geomPath = muscle.getElementsByTagName('GeometryPath').item(0);
        if isempty(geomPath), continue; end
        pathPointSet = geomPath.getElementsByTagName('PathPointSet').item(0);
        if isempty(pathPointSet), continue; end
        allPP = pathPointSet.getElementsByTagName('PathPoint');
        
        for pi = 0:allPP.getLength-1
            pp = allPP.item(pi);
            if isempty(pp), continue; end
            pp_name = char(pp.getAttribute('name'));
            
            idx_dash = strfind(pp_name, '-');
            if isempty(idx_dash), continue; end
            baseName = pp_name(1:idx_dash(end)-1);
            suffix = pp_name(idx_dash(end):end);
            
            row_idx = find(strcmp(muscle_table(:,1), baseName) & strcmp(muscle_table(:,2), suffix), 1);
            if isempty(row_idx), continue; end
            
            coords = cell2mat(muscle_table(row_idx, 3:5));
            loc_node = pp.getElementsByTagName('location').item(0);
            if isempty(loc_node), continue; end
            loc_node.getFirstChild.setNodeValue(sprintf('%.6f %.6f %.6f', coords));
        end
    end
end

%% ================= GEOMETRY HELPER FUNCTIONS =================

function [centre, radius] = lsqSphere(P, r_bounds)
    % Least-squares sphere fitting with radius bounds
    % P: Nx3 point cloud
    % r_bounds: [min_radius, max_radius]
    
    c0 = mean(P,1)';
    r0 = mean(sqrt(sum((P - c0').^2, 2)));
    x0 = [c0; r0];
    fun = @(x) sqrt(sum((P - x(1:3)').^2, 2)) - x(4);
    lb = [-Inf; -Inf; -Inf; r_bounds(1)];
    ub = [ Inf;  Inf;  Inf; r_bounds(2)];
    options = optimoptions('lsqnonlin', 'Display', 'off');
    x_opt = lsqnonlin(fun, x0, lb, ub, options);
    centre = x_opt(1:3);
    radius = x_opt(4);
end

function [n, origin] = fitPlaneNormal(P)
    % Fit a plane to point cloud and return the normal vector
    % Uses SVD to find the least-squares plane
    
    centroid = mean(P,1);
    [~,~,V] = svd(bsxfun(@minus, P, centroid), 0);
    n = V(:,3);  % Normal vector is the singular vector with smallest singular value
    if n(3) < 0, n = -n; end  % Orient upward
    origin = centroid';
end

function eul = normal2eulerXYZ_constrained(n, plane_normal)
    % Convert plane normal to XYZ Euler angles with reference plane constraint
    % n: normal vector (becomes Z-axis)
    % plane_normal: reference plane normal for constraining X-axis orientation
    
    n = n(:);
    plane_normal = plane_normal(:);
    z_axis = n / norm(n);
    
    % Determine reference direction
    ref = [1; 0; 0];
    if abs(dot(ref, plane_normal)) > 0.9
        ref = [0; 1; 0];
    end
    
    y_temp = ref - dot(ref, plane_normal) * plane_normal;
    y_temp = y_temp / norm(y_temp);
    y_axis = y_temp - dot(y_temp, z_axis) * z_axis;
    y_axis = y_axis / norm(y_axis);
    x_axis = cross(y_axis, z_axis);
    x_axis = x_axis / norm(x_axis);
    y_axis = cross(z_axis, x_axis);
    y_axis = y_axis / norm(y_axis);
    
    R = [x_axis, y_axis, z_axis];
    eul = [atan2(-R(2,3), R(3,3)); asin(R(1,3)); atan2(-R(1,2), R(1,1))];
end

function eul = normal2eulerXYZ_withHip(n, hip_center, plane_origin)
    % Convert plane normal to XYZ Euler angles using hip joint direction
    % n: normal vector (becomes Z-axis)
    % hip_center: hip joint center location
    % plane_origin: origin of the plane
    
    n = n(:);
    hip_center = hip_center(:);
    plane_origin = plane_origin(:);
    z_axis = n / norm(n);
    
    v = plane_origin - hip_center;
    v_proj = v - dot(v, z_axis) * z_axis;
    v_proj = v_proj / norm(v_proj);
    y_axis = v_proj;
    x_axis = cross(y_axis, z_axis);
    x_axis = x_axis / norm(x_axis);
    y_axis = cross(z_axis, x_axis);
    y_axis = y_axis / norm(y_axis);
    
    R = [x_axis, y_axis, z_axis];
    eul = [atan2(-R(2,3), R(3,3)); asin(R(1,3)); atan2(-R(1,2), R(1,1))];
end

%% ================= PUBIC ORIENTATION FUNCTIONS =================

function eul = computePubicOrientationFromPlane(PS_l_pts, plane_normal)
    % Compute pubic orientation from pelvis reference plane
    % Output: XYZ Euler angles in radians
    
    plane_normal = plane_normal(:);
    plane_normal = plane_normal / norm(plane_normal);
    
    % Y-axis: project plane normal onto XY plane
    y_axis = plane_normal;
    y_axis(3) = 0;
    if norm(y_axis) < 1e-8
        warning('plane_normal projection degenerate, using y=[0 1 0]');
        y_axis = [0;1;0];
    end
    y_axis = y_axis / norm(y_axis);
    
    % X-axis: orthogonal to Y-axis in XY plane
    global_z = [0;0;1];
    x_axis = cross(y_axis, global_z);
    if norm(x_axis) < 1e-8
        warning('x_axis degenerate, using x=[1 0 0]');
        x_axis = [1;0;0];
    end
    x_axis = x_axis / norm(x_axis);
    
    % Z-axis: complete right-handed system
    z_axis = cross(x_axis, y_axis);
    z_axis = z_axis / norm(z_axis);
    
    % Build rotation matrix with 90-degree Z-rotation
    R = [x_axis, y_axis, z_axis];
    Rz = [0 -1 0; 1 0 0; 0 0 1];
    R = R * Rz;
    
    % Convert to XYZ Euler angles
    eul = [atan2(-R(2,3), R(3,3)); asin(R(1,3)); atan2(-R(1,2), R(1,1))];
end

function eul = computePubicOrientationFromLongAxis(PS_l_pts)
    % Compute pubic orientation from long axis projected onto XY plane
    % Input: PS_l_pts - left pubic symphysis point cloud (Nx3)
    % Output: eul - XYZ Euler angles in radians
    
    % Compute long axis direction (line connecting two farthest points)
    D = pdist2(PS_l_pts, PS_l_pts);
    [~, maxIdx] = max(D(:));
    [i1, i2] = ind2sub(size(D), maxIdx);
    long_axis = PS_l_pts(i2,:) - PS_l_pts(i1,:);
    long_axis = long_axis / norm(long_axis);
    
    % Project onto XY plane (ignore Z component)
    a = long_axis;
    a(3) = 0;
    if norm(a) < 1e-8
        [~,~,Vps] = svd(bsxfun(@minus, PS_l_pts, mean(PS_l_pts,1)), 0);
        pc1 = Vps(:,1)';
        a = pc1; a(3) = 0;
        if norm(a) < 1e-8
            warning('Pubic long axis projection degenerate, using X-axis as default');
            a = [1 0 0];
        end
    end
    a = a / norm(a);
    
    % Construct right-handed coordinate system (Z-axis upward)
    z_axis = [0; 0; 1];
    x_axis = -a(:);
    y_axis = cross(z_axis, x_axis);
    y_axis = y_axis / norm(y_axis);
    x_axis = cross(y_axis, z_axis);
    x_axis = x_axis / norm(x_axis);
    
    % Convert rotation matrix to XYZ Euler angles
    R = [x_axis, y_axis, z_axis];
    eul = [atan2(-R(2,3), R(3,3)); asin(R(1,3)); atan2(-R(1,2), R(1,1))];
end

function modifyJointWithOrientation(xDoc, joint_name, coord, offset_name, eul)
    % Modify both translation and orientation of a joint's offset frame
    % For L5_S1_IVDjnt, also updates L56_offset with the same orientation
    
    jointSet = xDoc.getElementsByTagName('JointSet').item(0);
    joints = jointSet.getElementsByTagName('CustomJoint');
    
    % Find target joint
    joint = [];
    for j = 0:joints.getLength-1
        jnt = joints.item(j);
        if strcmp(char(jnt.getAttribute('name')), joint_name)
            joint = jnt; break;
        end
    end
    if isempty(joint), return; end
    
    frames = joint.getElementsByTagName('frames').item(0);
    offsets = frames.getElementsByTagName('PhysicalOffsetFrame');
    
    for k = 0:offsets.getLength-1
        frame = offsets.item(k);
        frame_name = char(frame.getAttribute('name'));
        
        if strcmp(frame_name, offset_name)
            % Update translation of the specified offset frame
            trans = frame.getElementsByTagName('translation').item(0);
            if isempty(trans)
                trans = xDoc.createElement('translation');
                frame.appendChild(trans);
            end
            trans.getFirstChild.setNodeValue(sprintf('%.6f %.6f %.6f', coord));
            
            % Update orientation
            orient = frame.getElementsByTagName('orientation').item(0);
            orient_str = sprintf('%.6f %.6f %.6f', eul);
            if isempty(orient)
                orient = xDoc.createElement('orientation');
                orient_text = xDoc.createTextNode(orient_str);
                orient.appendChild(orient_text);
                frame.appendChild(orient);
            else
                orient.getFirstChild.setNodeValue(orient_str);
            end
            
        elseif strcmp(frame_name, 'L56_offset') && strcmp(joint_name, 'L5_S1_IVDjnt')
            % Also update L56_offset orientation (same value as sacrum_offset)
            orient = frame.getElementsByTagName('orientation').item(0);
            orient_str = sprintf('%.6f %.6f %.6f', eul);
            if isempty(orient)
                orient = xDoc.createElement('orientation');
                orient_text = xDoc.createTextNode(orient_str);
                orient.appendChild(orient_text);
                frame.appendChild(orient);
            else
                orient.getFirstChild.setNodeValue(orient_str);
            end
        end
    end
end

function eul = normal2eulerXYZ_L5S1_unified(n, pelvis_x_axis)
    % Unified L5/S1 orientation computation: XZ plane fitting with horizontal X-axis
    % aligned with pelvis X-axis
    % Input: n - plane normal vector from point cloud fitting (Y-axis direction)
    %        pelvis_x_axis - pelvis X-axis direction (horizontal, pointing anteriorly)
    % Output: eul - XYZ Euler angles in radians
    
    n = n(:);
    pelvis_x_axis = pelvis_x_axis(:);
    
    % Y-axis = plane normal vector
    y_axis = n / norm(n);
    
    % Ensure Y-axis points upward (positive Z)
    if y_axis(3) < 0
        y_axis = -y_axis;
    end
    
    % X-axis: project pelvis X-axis onto horizontal plane (perpendicular to Y-axis)
    x_axis = pelvis_x_axis - dot(pelvis_x_axis, y_axis) * y_axis;
    if norm(x_axis) < 1e-8
        % Fallback to default X-axis if degenerate
        x_axis = [1;
