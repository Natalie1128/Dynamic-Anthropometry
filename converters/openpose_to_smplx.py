"""
Batch Assemble SMPL-X Keypoints

This script converts OpenPose keypoints stored in JSON files into SMPL-X-style
2D keypoints format. It reads keypoints from JSON files located in the INPUT_DIR,
maps the 2D keypoints from body, left/right hands, and face to a new order, and
saves the result as new JSON files in the OUTPUT_DIR.

Mapping Details:
    - 'body_map' defines the remapping of 21 body keypoints.
    - 'left_hand_map' and 'right_hand_map' remap the 15 keypoints for each hand.
    - 'face_map' remaps four selected face keypoints.

Usage:
   python batch_assemble_smplx_keypoints.py
"""

import json
import os
from pathlib import Path

INPUT_DIR = Path(r'C:\Users\natalie.n.griffin4\Documents\DA\squat\openpose_keypoints')
OUTPUT_DIR = Path(r'C:\Users\natalie.n.griffin4\Documents\DA\squat\smplx_keypoints')

#Define keypoints mapping dictionaries.
body_map = {
    0: 8,  1: 12, 2: 9,  3: 1,  4: 13,
    5: 10, 6: 0,  7: 14, 8: 11, 9: 1,
    10: 19, 11: 22, 12: 1, 13: 5, 14: 2,
    15: 0,  16: 6,  17: 3, 18: 7, 19: 4, 20: 20
}
left_hand_map = {i: i - 21 for i in range(21, 36)}
right_hand_map = {i: i - 36 for i in range(36, 51)}
face_map = {51: 43, 52: 37, 53: 18, 54: 23}

os.makedirs(OUTPUT_DIR, exist_ok=True)

for json_file in sorted(INPUT_DIR.glob('*.json')):
    data = json.load(json_file.open())
    p = data['people'][0]
    smplx = []

    body = p['pose_keypoints_2d']
    lh = p['hand_left_keypoints_2d']
    rh = p['hand_right_keypoints_2d']
    face = p['face_keypoints_2d']

    #Assemble body keypoints.
    for j in range(21):
        smplx += body[body_map[j]*3 : body_map[j]*3+3]
    #Assemble left hand keypoints.
    for j in range(21, 36):
        smplx += lh[left_hand_map[j]*3 : left_hand_map[j]*3+3]
    #Assemble right hand keypoints.
    for j in range(36, 51):
        smplx += rh[right_hand_map[j]*3 : right_hand_map[j]*3+3]
    #Assemble selected face keypoints.
    for j in range(51, 55):
        smplx += face[face_map[j]*3 : face_map[j]*3+3]

    base = json_file.stem.replace('_openpose_keypoints', '')
    out_file = OUTPUT_DIR / f"{base}_smplx_keypoints.json"
    
    json.dump({'smplx_keypoints_2d': smplx}, out_file.open('w'), indent=2)
    print(f"Converted {json_file.name} → {out_file.name}")
