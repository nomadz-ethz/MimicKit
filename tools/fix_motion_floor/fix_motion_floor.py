"""
NOTE: MADE FOR EXPERIMENT, AS FOOT GROUND PENETRATION MESSES K1 TRAINING. DOES NOT WORK. USE AS POSSIBLE IDEA ONLY. WE NEED BETTER REFERENCE MOTION PIPELINE.

Floor-snapping post-processor for MimicKit motion files.

Computes forward kinematics for every frame using the robot's kinematic model,
finds the lowest contact-body surface z-position across all frames (using the
body's world rotation to project the local geom offset), and shifts the root
z-coordinate so that the lowest point sits at z = -margin (default 0).

A small positive margin (e.g. 0.01 m) makes the floor level sit 1 cm below
the minimum contact surface, so every frame spawns with feet at or slightly
into the ground — avoiding the "spawns too high and falls" problem that occurs
when the minimum is an outlier frame.

Usage (from repo root):
    python tools/fix_motion_floor/fix_motion_floor.py \
        --char_file  data/assets/k1/k1.xml \
        --motion_file data/motions/k1/kick_soogon_2_mk.pkl \
        --foot_bodies left_foot_link right_foot_link \
        --geom_offset 0.026 0.0 -0.038 \
        --margin 0.01
"""

import argparse
import os
import sys
import numpy as np
import torch

# Allow imports from mimickit/ using the same convention as run.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mimickit"))

from anim.mjcf_char_model import MJCFCharModel
from anim.motion import load_motion, Motion
from util.torch_util import exp_map_to_quat, quat_pos, quat_rotate


def compute_floor_offset(char_file, motion_file, foot_body_names, geom_offset_local, margin):
    """
    Returns the z-offset that must be added to every frame's root_pos[2].

    The offset is chosen so that:
        min_over_frames( contact_surface_z ) == -margin

    where contact_surface_z is computed rotation-aware:
        contact_surface_z[frame] = body_pos_z + (R_world @ geom_offset_local)[2]

    Parameters
    ----------
    char_file : str
        Path to the MJCF robot XML.
    motion_file : str
        Path to the MimicKit motion pickle.
    foot_body_names : list[str]
        Body names whose contact surface determines the floor level.
    geom_offset_local : array-like, shape (3,)
        Vector from the body origin to the lowest contact point in the body's
        local frame (e.g. [0.026, 0, -0.038] for K1's foot collision box).
    margin : float
        Extra downward bias in metres.  Positive = robot spawns slightly into
        the ground (physics resolves it gently), preventing the "too high" problem.

    Returns
    -------
    float
        z-offset to add to root_pos[:, 2].
    """
    device = "cpu"

    # Load kinematic model
    kin_model = MJCFCharModel(device=device)
    kin_model.load(char_file)

    # Load motion frames  shape: (N, 6 + num_dofs)
    motion_data = load_motion(motion_file)
    frames = torch.tensor(motion_data.frames, dtype=torch.float32)  # (N, D)
    N = frames.shape[0]

    root_pos    = frames[:, 0:3]   # (N, 3)
    root_expmap = frames[:, 3:6]   # (N, 3)
    joint_dof   = frames[:, 6:]    # (N, num_dofs)

    root_rot_quat = quat_pos(exp_map_to_quat(root_expmap))  # (N, 4)
    joint_rot     = quat_pos(kin_model.dof_to_rot(joint_dof))  # (N, J, 4)

    # Forward kinematics → world positions AND rotations for every body
    body_pos, body_rot = kin_model.forward_kinematics(root_pos, root_rot_quat, joint_rot)
    # body_pos: (N, num_bodies, 3)
    # body_rot: (N, num_bodies, 4)  quaternions xyzw

    geom_local = torch.tensor(geom_offset_local, dtype=torch.float32)  # (3,)
    geom_local_batch = geom_local.unsqueeze(0).expand(N, 3)            # (N, 3)

    min_z = float("inf")
    for name in foot_body_names:
        bid = kin_model.get_body_id(name)
        body_z   = body_pos[:, bid, 2]   # (N,)
        body_r   = body_rot[:, bid, :]   # (N, 4)

        # Rotate geom offset into world frame, then take z component
        geom_world = quat_rotate(body_r, geom_local_batch)   # (N, 3)
        contact_z  = body_z + geom_world[:, 2]               # (N,)

        frame_min = contact_z.min().item()
        print(f"  [{name}]  contact-surface z:  min={frame_min:.4f}  "
              f"max={contact_z.max().item():.4f}  mean={contact_z.mean().item():.4f}")
        min_z = min(min_z, frame_min)

    print(f"  Global minimum contact-surface z: {min_z:.4f} m")
    print(f"  Margin: {margin:.4f} m  →  target floor = {-margin:.4f} m")

    # Shift so that min contact_z == -margin  (i.e. add offset = -min_z - margin)
    offset = -min_z - margin
    return offset


def apply_floor_offset(motion_file, output_file, offset):
    """Load motion, shift root z by offset, save to output_file."""
    motion_data = load_motion(motion_file)
    frames = np.array(motion_data.frames, dtype=np.float32)

    print(f"  Root z before: min={frames[:, 2].min():.4f}  max={frames[:, 2].max():.4f}")
    frames[:, 2] += offset
    print(f"  Root z after:  min={frames[:, 2].min():.4f}  max={frames[:, 2].max():.4f}")

    corrected = Motion(loop_mode=motion_data.loop_mode,
                       fps=motion_data.fps,
                       frames=frames)
    corrected.save(output_file)
    print(f"  Saved corrected motion to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Floor-snap a MimicKit motion file using rotation-aware forward kinematics."
    )
    parser.add_argument("--char_file",    required=True,
                        help="Path to the robot MJCF XML (e.g. data/assets/k1/k1.xml)")
    parser.add_argument("--motion_file",  required=True,
                        help="Path to the input MimicKit motion pickle")
    parser.add_argument("--output_file",  default=None,
                        help="Path for corrected output (default: overwrites input)")
    parser.add_argument("--foot_bodies",  nargs="+",
                        default=["left_foot_link", "right_foot_link"],
                        help="Body names whose contact surface sets the floor level")
    parser.add_argument("--geom_offset",  nargs=3, type=float,
                        default=[0.026, 0.0, -0.038],
                        metavar=("X", "Y", "Z"),
                        help="Local-frame vector from body origin to contact surface "
                             "(default: K1 foot box bottom  0.026 0 -0.038)")
    parser.add_argument("--margin",       type=float, default=0.01,
                        help="Extra downward bias in metres (default 0.01). "
                             "Positive = robot spawns feet slightly into the ground, "
                             "avoiding the 'spawns too high' problem.")
    args = parser.parse_args()

    output_file = args.output_file if args.output_file else args.motion_file

    print("=" * 60)
    print("Floor-snapping motion file")
    print(f"  char_file:    {args.char_file}")
    print(f"  motion_file:  {args.motion_file}")
    print(f"  output_file:  {output_file}")
    print(f"  foot_bodies:  {args.foot_bodies}")
    print(f"  geom_offset:  {args.geom_offset}")
    print(f"  margin:       {args.margin} m")
    print("=" * 60)

    offset = compute_floor_offset(
        char_file=args.char_file,
        motion_file=args.motion_file,
        foot_body_names=args.foot_bodies,
        geom_offset_local=args.geom_offset,
        margin=args.margin,
    )
    print(f"  Computed z-offset to apply: {offset:+.4f} m")

    apply_floor_offset(args.motion_file, output_file, offset)

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
