%% ================= Modify SIJ and Pubic Joint Motion Axes in OpenSim Model =================
% Automatically read three pelvic ring points from OSIM file, compute normal vector,
% transform to each joint's local coordinate system, and modify the axis
clear; clc;

%% ================= Configuration =================
% OSIM model file path (modify to actual path)
osim_file = 'J:\PG_Pelvis\WholePelvis\PCA_predict\output\split_pelvis\import_pelvis_into_model\New_p019_AnatFrame.osim';

% Check if file exists
if ~exist(osim_file, 'file')
    error('File does not exist: %s', osim_file);
end

% Output file path (append _FuncFrame suffix)
[filepath, filename, ext] = fileparts(osim_file);
output_file = fullfile(filepath, [filename, '_FuncFrame', ext]);
fprintf('Input file: %s\n', osim_file);
fprintf('Output file: %s\n\n', output_file);

%% ================= Read OSIM File =================
xDoc = xmlread(osim_file);

%% ================= Automatically Read Three Pelvic Ring Point Coordinates =================
% Get JointSet
jointSet = xDoc.getElementsByTagName('JointSet').item(0);
joints = jointSet.getElementsByTagName('CustomJoint');

% Initialize storage variables
pubic_point = [];
sacrumjnt_l_point = [];
sacrumjnt_r_point = [];

% Store each joint's orientation (for subsequent coordinate transformation)
joint_orientations = struct();

% Iterate over all CustomJoints
for j = 0:joints.getLength-1
    joint = joints.item(j);
    joint_name = char(joint.getAttribute('name'));
    
    % Get frames under PhysicalOffsetFrame
    frames = joint.getElementsByTagName('frames').item(0);
    if isempty(frames)
        continue;
    end
    
    offsets = frames.getElementsByTagName('PhysicalOffsetFrame');
    for k = 0:offsets.getLength-1
        offset = offsets.item(k);
        offset_name = char(offset.getAttribute('name'));
        
        % Only process pelvis_l_offset and pelvis_r_offset (which contain translation and orientation)
        if ~strcmp(offset_name, 'pelvis_l_offset') && ~strcmp(offset_name, 'pelvis_r_offset')
            continue;
        end
        
        % Read translation
        trans_node = offset.getElementsByTagName('translation').item(0);
        if ~isempty(trans_node)
            trans_str = char(trans_node.getFirstChild.getData());
            trans = sscanf(trans_str, '%f %f %f');
            trans = trans(:)';  % Convert to row vector
        else
            trans = [0, 0, 0];
        end
        
        % Read orientation
        orient_node = offset.getElementsByTagName('orientation').item(0);
        if ~isempty(orient_node)
            orient_str = char(orient_node.getFirstChild.getData());
            orient = sscanf(orient_str, '%f %f %f');
            orient = orient(:)';  % Convert to row vector
        else
            orient = [0, 0, 0];
        end
        
        % Store orientation (keyed by joint name and offset name)
        key = [joint_name, '_', offset_name];
        joint_orientations.(key) = orient;
        
        % Determine point type based on joint name and offset name
        if contains(joint_name, 'pubic') && (strcmp(offset_name, 'pelvis_l_offset') || strcmp(offset_name, 'pelvis_r_offset'))
            pubic_point = trans;
            fprintf('Read pubic point (from %s/%s): [%.6f, %.6f, %.6f]\n', joint_name, offset_name, trans);
        elseif strcmp(joint_name, 'sacrumjnt_l') && strcmp(offset_name, 'pelvis_l_offset')
            sacrumjnt_l_point = trans;
            fprintf('Read sacrumjnt_l point: [%.6f, %.6f, %.6f]\n', trans);
        elseif strcmp(joint_name, 'sacrumjnt_r') && strcmp(offset_name, 'pelvis_r_offset')
            sacrumjnt_r_point = trans;
            fprintf('Read sacrumjnt_r point: [%.6f, %.6f, %.6f]\n', trans);
        end
    end
end

% Check if all three points were successfully read
if isempty(pubic_point)
    error('Pubic point translation not found');
end
if isempty(sacrumjnt_l_point)
    error('sacrumjnt_l point translation not found');
end
if isempty(sacrumjnt_r_point)
    error('sacrumjnt_r point translation not found');
end

fprintf('\nPelvic ring three-point coordinates:\n');
fprintf('  pubic:        [%.6f, %.6f, %.6f]\n', pubic_point);
fprintf('  sacrumjnt_l:  [%.6f, %.6f, %.6f]\n', sacrumjnt_l_point);
fprintf('  sacrumjnt_r:  [%.6f, %.6f, %.6f]\n', sacrumjnt_r_point);

%% ================= Step 1: Compute Pelvic Ring Plane Normal Vector (Global Coordinate System) =================
v1 = sacrumjnt_l_point - pubic_point;
v2 = sacrumjnt_r_point - pubic_point;

% Compute normal vector
n_global = cross(v1, v2);
n_global = n_global / norm(n_global);  % Normalize

fprintf('\nPelvic ring plane normal vector (global coordinate system): [%.6f, %.6f, %.6f]\n', n_global);

%% ================= Step 2: Get Each Joint's Local Coordinate System Orientation =================
% Define joints to process and their corresponding offsets
joints_to_process = {
    'sacrumjnt_l', 'pelvis_l_offset', 'rotation1';   % SIJ left, modify rotation1
    'sacrumjnt_r', 'pelvis_r_offset', 'rotation1';   % SIJ right, modify rotation1
    'pubic_l',     'pelvis_l_offset', 'rotation2';   % Pubic left, modify rotation2
    'pubic_r',     'pelvis_r_offset', 'rotation2';   % Pubic right, modify rotation2
};

fprintf('\nComputing normal vector in each joint local coordinate system:\n');

% Preallocate storage
joint_updates = cell(size(joints_to_process, 1), 3);

for i = 1:size(joints_to_process, 1)
    joint_name = joints_to_process{i, 1};
    offset_name = joints_to_process{i, 2};
    transform_axis_name = joints_to_process{i, 3};
    
    % Get orientation
    key = [joint_name, '_', offset_name];
    if ~isfield(joint_orientations, key)
        warning('Orientation for %s not found, using default [0, 0, 0]', key);
        orient = [0, 0, 0];
    else
        orient = joint_orientations.(key);
    end
    
    % Convert Euler angles to rotation matrix (call local function)
    R_parent_to_child = eul2rotm_xyz(orient);
    
    % Transform global normal vector to local coordinate system
    n_local = R_parent_to_child' * n_global(:);
    n_local = n_local / norm(n_local);
    
    % Store result
    joint_updates{i, 1} = joint_name;
    joint_updates{i, 2} = transform_axis_name;
    joint_updates{i, 3} = n_local;
    
    fprintf('  %s: local normal vector = [%.6f, %.6f, %.6f]\n', joint_name, n_local(1), n_local(2), n_local(3));
end

%% ================= Step 3: Modify <axis> in OSIM File =================
% Re-acquire JointSet (ensure it is up to date)
jointSet = xDoc.getElementsByTagName('JointSet').item(0);
joints = jointSet.getElementsByTagName('CustomJoint');

fprintf('\nModifying joint <axis>:\n');

for i = 1:size(joint_updates, 1)
    joint_name = joint_updates{i, 1};
    transform_axis_name = joint_updates{i, 2};
    n_local = joint_updates{i, 3};
    
    % Find the corresponding joint
    for j = 0:joints.getLength-1
        joint = joints.item(j);
        if strcmp(char(joint.getAttribute('name')), joint_name)
            % Get SpatialTransform
            spatialTransform = joint.getElementsByTagName('SpatialTransform').item(0);
            if isempty(spatialTransform)
                warning('SpatialTransform for %s not found', joint_name);
                break;
            end
            
            % Get all TransformAxes
            transformAxes = spatialTransform.getElementsByTagName('TransformAxis');
            
            % Find the corresponding TransformAxis by name (rotation1 or rotation2)
            for k = 0:transformAxes.getLength-1
                ta = transformAxes.item(k);
                ta_name = char(ta.getAttribute('name'));
                if strcmp(ta_name, transform_axis_name)
                    % Modify the value of the <axis> tag
                    axis_node = ta.getElementsByTagName('axis').item(0);
                    if ~isempty(axis_node)
                        new_axis_value = sprintf('%.6f %.6f %.6f', n_local(1), n_local(2), n_local(3));
                        axis_node.getFirstChild.setNodeValue(new_axis_value);
                        fprintf('  ✅ %s / %s: axis = [%s]\n', joint_name, transform_axis_name, new_axis_value);
                    end
                    break;
                end
            end
            break;
        end
    end
end

%% ================= Step 4: Save Modified Model =================
xmlwrite(output_file, xDoc);
fprintf('\n✅ Model saved to: %s\n', output_file);
fprintf('   Original file: %s\n', osim_file);
fprintf('   New file: %s\n', output_file);

%% ================= Local Function Definition (must be placed at end of file) =================
function R = eul2rotm_xyz(eul)
    % XYZ Euler angles to rotation matrix
    % eul = [rx, ry, rz] in radians
    rx = eul(1);
    ry = eul(2);
    rz = eul(3);
    
    Rx = [1, 0, 0;
          0, cos(rx), -sin(rx);
          0, sin(rx), cos(rx)];
    
    Ry = [cos(ry), 0, sin(ry);
          0, 1, 0;
          -sin(ry), 0, cos(ry)];
    
    Rz = [cos(rz), -sin(rz), 0;
          sin(rz), cos(rz), 0;
          0, 0, 1];
    
    R = Rx * Ry * Rz;
end
