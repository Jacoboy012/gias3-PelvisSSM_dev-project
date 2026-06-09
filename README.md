This repository provides a complete workflow for generating a female pelvis Statistical Shape Model (SSM), reconstructing subject-specific pelvis geometry from motion capture markers, and importing the personalized pelvis into OpenSim models.

## Important Note

This workflow only personalizes and scales the pelvis geometry. It is **not a full-body scaling workflow**. The sizes and masses of all other body segments in the OpenSim model should either:

- be scaled beforehand using the standard OpenSim linear scaling approach, or
- be scaled after completing this workflow using the standard OpenSim linear scaling approach.

Only the pelvis geometry is reconstructed and personalized by the present workflow.

## Dataset

The pelvis SSM was developed using a dataset of:

150 CT scans of female participants
Age: 34.2 ± 8.4 years (range: 20–49)
Height: 163.3 ± 7.9 cm (range: 155–195)
Weight: 71.5 ± 20.4 kg (range: 43–166)

The generated pelvis SSM model is provided under the /data directory.

## Main Scripts

1. Statistical Shape Model Generation

Builds the pelvis Statistical Shape Model using Principal Component Analysis (PCA). Please check https://github.com/musculoskeletal/gias3 for more details.

2. Subject-Specific Pelvis Reconstruction from Motion Capture

run_pca_rig_markers.py: reconstructs a plausible subject-specific pelvis geometry based on relative positions of pelvis skin markers collected from motion capture, such as: Left / Right ASIS, Left / Right PSIS.The script uses the trained SSM to estimate anatomically realistic pelvis bone shape.

3. Pelvis Deviation

WholePelvis.split.py：splits the complete pelvis into: Left innominate bone, Right innominate bone, Sacrum.

4. Joint Landmarks

Joint and muscle landmarks were calculated and provided under the /data directory.

5. OpenSim Integration
   
import_pelvis_2osim.m: imports the personalized pelvis geometry into the OpenSim model.
