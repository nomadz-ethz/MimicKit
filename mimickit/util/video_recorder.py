from __future__ import annotations

import numpy as np
import os
import tempfile
from typing import TYPE_CHECKING, Any

from util.logger import Logger

if TYPE_CHECKING:
    import engines.engine as engine
    from omni.replicator.core import Annotator, RenderProduct


class VideoRecorder:
    """Records video frames from the simulation and uploads to WandB.
    
    Works with Isaac Lab engine in headless mode using the omni.replicator
    annotator API to capture viewport images.

    TODO: Add support for other engines beyond Isaac Lab.
    
    Args:
        engine: The simulation engine (e.g. IsaacLabEngine).
        video_length: Number of steps per video recording.
        video_interval: Interval (in training steps) between video recordings.
        resolution: Tuple (width, height) for the captured frames.
        fps: Frames per second for the output video.
        cam_prim_path: USD prim path for the camera to capture from.
    """

    def __init__(self, engine: engine.Engine, video_length: int = 200,
                 video_interval: int = 2000, resolution: tuple[int, int] = (640, 480),
                 fps: int = 30, cam_prim_path: str = "/OmniverseKit_Persp") -> None:
        self._engine: engine.Engine = engine
        self._video_length: int = video_length
        self._video_interval: int = video_interval
        self._resolution: tuple[int, int] = resolution
        self._fps: int = fps
        self._cam_prim_path: str = cam_prim_path

        self._recorded_frames: list[np.ndarray] = []
        self._recording: bool = False
        self._global_step: int = 0
        self._video_count: int = 0

        self._annotator: Any | None = None
        self._render_product: Any | None = None

        return

    def _ensure_annotator(self) -> None:
        """Lazily create the render product and RGB annotator."""
        if self._annotator is not None:
            return
        
        import omni.replicator.core as rep

        self._render_product = rep.create.render_product(
            self._cam_prim_path, self._resolution
        )
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._annotator.attach([self._render_product])
        Logger.print("[VideoRecorder] Created RGB annotator for {}".format(self._cam_prim_path))
        return

    def _capture_frame(self) -> None:
        """Capture a single RGB frame from the viewport."""
        self._ensure_annotator()

        # Render the scene to update the viewport
        self._engine._sim.render()

        rgb_data: Any = self._annotator.get_data()
        if rgb_data is None or rgb_data.size == 0:
            # Renderer still warming up
            frame: np.ndarray = np.zeros((self._resolution[1], self._resolution[0], 3), dtype=np.uint8)
        else:
            frame = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            frame = frame[:, :, :3]  # drop alpha channel

        self._recorded_frames.append(frame)
        return

    def pre_step(self) -> None:
        """Call before each training step to check if recording should start."""
        if not self._recording and (self._global_step % self._video_interval == 0):
            self._start_recording()
        return

    def post_step(self) -> None:
        """Call after each training step to capture frames and stop if done."""
        if self._recording:
            self._capture_frame()

            if len(self._recorded_frames) >= self._video_length:
                self._stop_recording()

        self._global_step += 1
        return

    def _start_recording(self) -> None:
        """Begin a new video recording."""
        self._recorded_frames = []
        self._recording = True
        Logger.print("[VideoRecorder] Started recording (step {})".format(self._global_step))
        return

    def _stop_recording(self) -> None:
        """Stop recording, create video, upload to WandB, and clean up."""
        if not self._recording or len(self._recorded_frames) == 0:
            self._recording = False
            return

        self._recording = False

        try:
            import wandb
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

            clip: ImageSequenceClip = ImageSequenceClip(self._recorded_frames, fps=self._fps)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                temp_path: str = tmp.name

            clip.write_videofile(temp_path, logger=None)

            if wandb.run is not None:
                wandb.log({
                    "video": wandb.Video(temp_path, format="mp4"),
                    "video_step": self._global_step,
                })
                Logger.print("[VideoRecorder] Uploaded video to WandB ({} frames, step {})".format(
                    len(self._recorded_frames), self._global_step))
            else:
                Logger.print("[VideoRecorder] WandB not initialized, skipping upload")

            # Clean up temp file
            os.remove(temp_path)

        except ImportError as e:
            Logger.print("[VideoRecorder] Missing dependency: {}. Video not saved.".format(e))
        except Exception as e:
            Logger.print("[VideoRecorder] Error creating video: {}".format(e))

        self._recorded_frames = []
        self._video_count += 1
        return

    def flush(self) -> None:
        """Force stop and upload any in-progress recording."""
        if self._recording:
            self._stop_recording()
        return
