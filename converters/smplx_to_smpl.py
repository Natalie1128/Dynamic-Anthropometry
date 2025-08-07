"""
SMPLX to SMPL Keypoint Converter

This script converts SMPLX format keypoint data to the simpler SMPL format.

SMPLX format contains 55 keypoints (165 values with x,y,confidence for each):
- 21 body keypoints
- 15 left hand keypoints
- 15 right hand keypoints
- 4 face keypoints

SMPL format uses a subset of 24 keypoints (72 values with x,y,confidence for each).
This script maps the appropriate keypoints from SMPLX to create SMPL keypoints.

"""

import json
from pathlib import Path

#Directory configuration
INPUT_DIR = Path(r"C:\Users\natalie.n.griffin4\Documents\DA\squat\smplx_keypoints")
OUTPUT_DIR = Path(r"C:\Users\natalie.n.griffin4\Documents\DA\squat\smpl_keypoints")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

#Mapping dictionary: SMPL index to SMPLX index
smpl_to_smplx_index_map = {
     0:  0,    # pelvis
     1:  1,    # L_hip
     2:  2,    # R_hip
     3:  3,    # spine1
     4:  4,    # L_knee
     5:  5,    # R_knee
     6:  6,    # spine2
     7:  7,    # L_ankle
     8:  8,    # R_ankle
     9:  9,    # spine3
    10: 10,    # L_foot            
    11: 11,    # R_foot          
    12: 12,    # neck
    13: 13,    # L_collar / shoulder root
    14: 14,    # R_collar
    15: 15,    # head (nose)     
    16: 16,    # L_shoulder
    17: 17,    # R_shoulder
    18: 18,    # L_elbow
    19: 19,    # R_elbow
    20: 21,    # L_wrist           first joint in L-hand cluster
    21: 36,    # R_wrist           first joint in R-hand cluster
    22: 22,    # L_hand / palm     second joint in L-hand cluster
    23: 37     # R_hand / palm     second joint in R-hand cluster
}

try:
    #Process each SMPLX keypoint file in the input directory
    for smplx_file in sorted(INPUT_DIR.glob("*_smplx_keypoints.json")):
        data = json.load(open(smplx_file))
        smplx_keypoints = data.get("smplx_keypoints_2d", [])
        
        #check the expected format (55 keypoints × 3 values per keypoint)
        if len(smplx_keypoints) != 165:
            print(f"Skipping {smplx_file.name}: unexpected length {len(smplx_keypoints)}")
            continue


        smpl_keypoints = []
        for smpl_index in range(24):
            smplx_source_index = smpl_to_smplx_index_map[smpl_index] * 3
            smpl_keypoints.extend(smplx_keypoints[smplx_source_index:smplx_source_index+3])

        #Create output filename
        base_filename = smplx_file.stem.replace("_smplx_keypoints", "")
        output_file_path = OUTPUT_DIR / f"{base_filename}_smpl_keypoints.json"
        
        #Save the transformed keypoints to the output file
        with open(output_file_path, "w") as f:
            json.dump({"smpl_keypoints_2d": smpl_keypoints}, f, indent=2)
        print(f"Converted {smplx_file.name} → {output_file_path.name}")

except Exception as e:
    print(f"Error processing files: {e}")
