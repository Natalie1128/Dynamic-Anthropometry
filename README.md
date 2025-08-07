# Dynamic Anthropometry Project

This project focuses on dynamic anthropometry, utilizing various scripts to process and analyze 3D human body data. The project includes functionalities for converting keypoint formats, aligning 3D scans, and rendering 3D meshes from multiple viewpoints.

## Project Structure

```
dynamic-anthropometry
├── converters
│   ├── alpha_to_open.py          # Converts AlphaPose JSON data to OpenPose JSON format.
│   ├── convert_to_meters.py      # Converts mesh dimensions from millimeters to meters.
│   ├── openpose_to_smplx.py      # Converts OpenPose keypoints to SMPL-X-style 2D keypoints.
│   └── smplx_to_smpl.py          # Converts SMPLX format keypoints to SMPL format.
├── pose_estimation
│   ├── 3D_skeleton_fusion.py     # Reconstructs a fused 3D skeleton from 2D keypoints.
│   ├── ik_fit_joints_smplx_pose_demo.py # Fits SMPLX poses to 3D keypoints using inverse kinematics.
│   └── smpl_scan_alignment.py     # Aligns a SMPL model to a 3D scan.
├── rendering
│   └── mesh_multiview_render.py   # Renders images of a 3D mesh from multiple viewpoints.
└── test_scripts
    └── extract_smplkeypoints_alpha3d.py # Extracts the best SMPL keypoints from detection results.
```

## Usage

Each script is designed to be run independently and requires specific input files. Ensure that the necessary dependencies, such as Open3D and NumPy, are installed in your Python environment.

## Dependencies

- Python 3.x
- Open3D
- NumPy
- SciPy
- Matplotlib

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd dynamic-anthropometry
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Contribution

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.