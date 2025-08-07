"""
Module: extract_smplkeypoints_alpha3d
Description:
    This script processes an input JSON file containing detection results (e.g., from HybrIK).
    It selects the best detection (highest score) for each unique image and extracts the first 24 SMPL keypoints.
    Each set of keypoints is saved as a new JSON file in the specified output directory.
    
Usage:
    python extract_smplkeypoints_alpha3d.py --results <path/to/alphapose-results.json> --outdir <output_directory>
"""

import argparse, json, pathlib, numpy as np


def main(results, outdir):

    results_json = results   
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    #Load JSON detection data.
    detections = json.load(open(results_json))
    best = {}
    #Iterate through each detection and select the best score per image.
    for det in detections:
        img_id = det["image_id"]
        if img_id not in best or det["score"] > best[img_id]["score"]:
            best[img_id] = det

    print(f"Found {len(best)} unique images.")

    #Process the best detection for each image.
    for img_id, det in best.items():
        #HybrIK outputs 29 keypoints; take the first 24 joints.
        kps24 = np.asarray(det["keypoints"], float).reshape(-1, 3)[:24]
        flat  = kps24.ravel().tolist()

        
        out = { "smpl_keypoints_2d": flat }
        out_name = f"{pathlib.Path(img_id).stem}_smpl_keypoints.json"
        json.dump(out, open(outdir / out_name, "w"))

    print("✓  wrote keypoints to", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract SMPL keypoints from HybrIK detection results.")
    ap.add_argument("--results", required=True,
                    help="Path to the alphapose-results.json file (HybrIK run)")
    ap.add_argument("--outdir",  required=True,
                    help="Directory to store the per-image SMPL keypoint JSON files")
    main(**vars(ap.parse_args()))