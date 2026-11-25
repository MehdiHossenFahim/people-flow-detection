"""
People Flow Detection & Spatial Heatmap Pipeline
Modular, production-grade vision pipeline for real-time pedestrian tracking,
bidirectional virtual line counting, and spatial density heatmap generation.
"""

import argparse
import os
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

# Add safe globals for PyTorch 2.6+ model deserialization compatibility
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass


class LineCrossingCounter:
    """
    Manages dual virtual counting lines (tripwires) and tracks directional crossings
    with anti-double-counting guarantees.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        line_in_ratio: float = 0.40,
        line_out_ratio: float = 0.75,
    ):
        """
        Initialize line coordinates and counter states.

        Args:
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.
            line_in_ratio: Fractional y-position (from top) for the upper IN line.
            line_out_ratio: Fractional y-position (from top) for the lower OUT line.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.line_upper_y = int(frame_height * line_in_ratio)
        self.line_lower_y = int(frame_height * line_out_ratio)

        self.line_upper_start = (0, self.line_upper_y)
        self.line_upper_end = (frame_width, self.line_upper_y)

        self.line_lower_start = (0, self.line_lower_y)
        self.line_lower_end = (frame_width, self.line_lower_y)

        self.in_count: int = 0
        self.out_count: int = 0

        # Persistent track state
        self.track_last_y: Dict[int, int] = {}
        self.counted_in_ids: Set[int] = set()
        self.counted_out_ids: Set[int] = set()

    def update(
        self,
        tracker_ids: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Evaluate pedestrian positions against counting lines and update counts.

        Args:
            tracker_ids: Array of persistent tracker integer IDs.
            boxes_xyxy: Array of bounding boxes in [x1, y1, x2, y2] format.

        Returns:
            List of foot-anchor points (cx, cy) for all active tracks in this frame.
        """
        foot_points: List[Tuple[int, int]] = []

        if tracker_ids is None or len(tracker_ids) == 0:
            return foot_points

        for i in range(len(tracker_ids)):
            tid = int(tracker_ids[i])
            x1, y1, x2, y2 = boxes_xyxy[i]
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            foot_points.append((cx, cy))

            if tid in self.track_last_y:
                prev_cy = self.track_last_y[tid]

                # IN line logic: downward crossing through upper line
                if prev_cy < self.line_upper_y <= cy and tid not in self.counted_in_ids:
                    self.in_count += 1
                    self.counted_in_ids.add(tid)

                # OUT line logic: upward crossing through lower line
                if prev_cy > self.line_lower_y >= cy and tid not in self.counted_out_ids:
                    self.out_count += 1
                    self.counted_out_ids.add(tid)

            self.track_last_y[tid] = cy

        return foot_points

    def draw_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draw the IN and OUT counting lines with text labels onto the frame."""
        # Upper IN Line (Green)
        cv2.line(frame, self.line_upper_start, self.line_upper_end, (0, 255, 0), 2)
        cv2.putText(
            frame,
            "IN line (Upper)",
            (15, max(self.line_upper_y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Lower OUT Line (Red)
        cv2.line(frame, self.line_lower_start, self.line_lower_end, (0, 0, 255), 2)
        cv2.putText(
            frame,
            "OUT line (Lower)",
            (15, max(self.line_lower_y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Render a dashboard HUD overlay showing current IN and OUT counts."""
        # Semi-transparent background banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (230, 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Border for the banner
        cv2.rectangle(frame, (10, 10), (230, 85), (80, 80, 80), 1)

        # Count texts
        cv2.putText(
            frame,
            f"IN : {self.in_count}",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"OUT: {self.out_count}",
            (25, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame


class SpatialHeatmapAccumulator:
    """
    Accumulates foot-traffic presence into a high-precision Float32 matrix
    and synthesizes Gaussian-smoothed density heatmaps.
    """

    def __init__(
        self,
        frame_height: int,
        frame_width: int,
        stamp_radius: int = 15,
        blur_sigma: float = 20.0,
        colormap: int = cv2.COLORMAP_JET,
    ):
        """
        Initialize the density accumulator.

        Args:
            frame_height: Video frame height.
            frame_width: Video frame width.
            stamp_radius: Radius in pixels stamped per detected footprint.
            blur_sigma: Standard deviation for Gaussian smoothing.
            colormap: OpenCV colormap constant for pseudo-coloring.
        """
        self.height = frame_height
        self.width = frame_width
        self.stamp_radius = stamp_radius
        self.blur_sigma = blur_sigma
        self.colormap = colormap

        self.heat_matrix = np.zeros((frame_height, frame_width), dtype=np.float32)

    def accumulate(self, foot_points: List[Tuple[int, int]], weight: float = 1.0) -> None:
        """Stamp presence circles for all active foot-anchor points."""
        for cx, cy in foot_points:
            if 0 <= cx < self.width and 0 <= cy < self.height:
                cv2.circle(self.heat_matrix, (cx, cy), self.stamp_radius, weight, -1)

    def generate(
        self,
        reference_frame: Optional[np.ndarray] = None,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate smoothed color heatmap and optional reference overlay.

        Args:
            reference_frame: Baseline video frame for spatial contextual overlay.
            alpha: Blending weight for the heatmap overlay (0.0 to 1.0).

        Returns:
            Tuple of (pure_heatmap_bgr, overlay_heatmap_bgr).
        """
        # Apply 2D Gaussian filter
        heat_blur = cv2.GaussianBlur(
            self.heat_matrix,
            (0, 0),
            sigmaX=self.blur_sigma,
            sigmaY=self.blur_sigma,
        )

        # Min-max normalization to [0, 255] uint8
        heat_norm = cv2.normalize(
            heat_blur,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        # Apply OpenCV colormap
        heat_color = cv2.applyColorMap(heat_norm, self.colormap)

        # Synthesize overlay if reference frame is available
        heatmap_overlay = None
        if reference_frame is not None:
            heatmap_overlay = cv2.addWeighted(
                reference_frame,
                1.0 - alpha,
                heat_color,
                alpha,
                0,
            )

        return heat_color, heatmap_overlay


class PeopleFlowPipeline:
    """
    End-to-end pipeline coordinating detection, ByteTrack tracking,
    virtual line counting, and heatmap accumulation.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.30,
        line_in_ratio: float = 0.40,
        line_out_ratio: float = 0.75,
        blur_sigma: float = 20.0,
        stamp_radius: int = 15,
        overlay_alpha: float = 0.50,
        device: Optional[str] = None,
    ):
        self.weights = weights
        self.conf = conf
        self.line_in_ratio = line_in_ratio
        self.line_out_ratio = line_out_ratio
        self.blur_sigma = blur_sigma
        self.stamp_radius = stamp_radius
        self.overlay_alpha = overlay_alpha
        self.device = device

        print(f"[Pipeline] Loading YOLO model from {weights}...")
        self.model = YOLO(weights)
        if device:
            self.model.to(device)

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=3,
        )
        self.person_class_id = 0

    @staticmethod
    def ensure_source(source: str) -> str:
        """Resolve source path or download sample video if given URL or default."""
        sample_url = (
            "https://media.roboflow.com/supervision/video-examples/people-walking.mp4"
        )
        if source.startswith("http://") or source.startswith("https://"):
            local_filename = os.path.basename(source.split("?")[0]) or "input_stream.mp4"
            if not os.path.exists(local_filename):
                print(f"[Pipeline] Downloading stream from {source} -> {local_filename}...")
                urllib.request.urlretrieve(source, local_filename)
            return local_filename

        if not os.path.exists(source) and source == "people-walking.mp4":
            print(f"[Pipeline] Default source '{source}' not found. Downloading sample video...")
            urllib.request.urlretrieve(sample_url, source)
            print(f"[Pipeline] Download complete: {source}")

        if not os.path.exists(source):
            raise FileNotFoundError(f"Source video file not found: {source}")

        return source

    def run(
        self,
        source: str,
        output_path: Optional[str] = "people_flow_output.mp4",
        save_heatmap: bool = True,
        heatmap_overlay_path: str = "final_heatmap_overlay.jpg",
        heatmap_pure_path: str = "final_heatmap_pure.jpg",
        write_video: bool = True,
    ) -> Dict[str, float]:
        """
        Execute full pipeline over the video stream.

        Returns:
            Dictionary containing final counts and performance metrics.
        """
        resolved_source = self.ensure_source(source)
        video_info = sv.VideoInfo.from_video_path(resolved_source)

        print("[Pipeline] Stream Metadata:")
        print(f"  - Resolution: {video_info.width} x {video_info.height}")
        print(f"  - Frame Rate: {video_info.fps:.2f} FPS")
        print(f"  - Total Frames: {video_info.total_frames}")

        tracker = sv.ByteTrack(frame_rate=video_info.fps)
        counter = LineCrossingCounter(
            frame_width=video_info.width,
            frame_height=video_info.height,
            line_in_ratio=self.line_in_ratio,
            line_out_ratio=self.line_out_ratio,
        )
        heatmap = SpatialHeatmapAccumulator(
            frame_height=video_info.height,
            frame_width=video_info.width,
            stamp_radius=self.stamp_radius,
            blur_sigma=self.blur_sigma,
        )

        reference_frame = None
        frame_generator = sv.get_video_frames_generator(resolved_source)

        start_time = time.time()
        processed_frames = 0

        # Video writer context
        if write_video and output_path:
            sink = sv.VideoSink(target_path=output_path, video_info=video_info)
            sink.__enter__()
        else:
            sink = None

        print("[Pipeline] Starting video processing...")
        try:
            for frame in frame_generator:
                if reference_frame is None:
                    reference_frame = frame.copy()

                # 1. Object Detection (YOLOv8)
                results = self.model(frame, conf=self.conf, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[detections.class_id == self.person_class_id]

                # 2. Tracking (ByteTrack)
                detections = tracker.update_with_detections(detections)

                # 3. Virtual Line Crossing & Counting
                if detections.tracker_id is not None and len(detections) > 0:
                    foot_points = counter.update(
                        tracker_ids=detections.tracker_id,
                        boxes_xyxy=detections.xyxy,
                    )
                    # 4. Spatial Heatmap Accumulation
                    heatmap.accumulate(foot_points)

                # 5. Visual Annotations & HUD
                if sink is not None:
                    annotated_frame = frame.copy()
                    counter.draw_lines(annotated_frame)

                    if detections.tracker_id is not None and len(detections) > 0:
                        labels = [f"ID {tid}" for tid in detections.tracker_id]
                        annotated_frame = self.box_annotator.annotate(
                            annotated_frame, detections
                        )
                        annotated_frame = self.label_annotator.annotate(
                            annotated_frame, detections, labels=labels
                        )

                    counter.draw_hud(annotated_frame)
                    sink.write_frame(annotated_frame)

                processed_frames += 1
                if processed_frames % 50 == 0 or processed_frames == video_info.total_frames:
                    elapsed = time.time() - start_time
                    fps = processed_frames / elapsed if elapsed > 0 else 0
                    print(
                        f"  [Frame {processed_frames:4d}/{video_info.total_frames}] "
                        f"FPS: {fps:5.1f} | IN: {counter.in_count:2d} | OUT: {counter.out_count:2d}"
                    )

        finally:
            if sink is not None:
                sink.__exit__(None, None, None)

        total_time = time.time() - start_time
        avg_fps = processed_frames / total_time if total_time > 0 else 0

        print("[Pipeline] Processing Complete!")
        print(f"  - Elapsed Time: {total_time:.2f} s")
        print(f"  - Average Throughput: {avg_fps:.1f} FPS")
        print(f"  - Final Counts: IN = {counter.in_count}, OUT = {counter.out_count}")

        # Heatmap export
        if save_heatmap:
            pure_heat, overlay_heat = heatmap.generate(
                reference_frame=reference_frame,
                alpha=self.overlay_alpha,
            )
            os.makedirs(os.path.dirname(heatmap_pure_path) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(heatmap_overlay_path) or ".", exist_ok=True)

            cv2.imwrite(heatmap_pure_path, pure_heat)
            if overlay_heat is not None:
                cv2.imwrite(heatmap_overlay_path, overlay_heat)

            print(f"[Pipeline] Saved pure heatmap to: {heatmap_pure_path}")
            print(f"[Pipeline] Saved heatmap overlay to: {heatmap_overlay_path}")

        if write_video and output_path:
            print(f"[Pipeline] Saved annotated video to: {output_path}")

        return {
            "in_count": counter.in_count,
            "out_count": counter.out_count,
            "total_frames": processed_frames,
            "avg_fps": avg_fps,
            "elapsed_seconds": total_time,
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="People Flow Detection & Spatial Density Heatmap CLI Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="people-walking.mp4",
        help="Path to input video file or remote URL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="people_flow_output.mp4",
        help="Path for saving the annotated output video.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold for YOLOv8 pedestrian detection.",
    )
    parser.add_argument(
        "--save-heatmap",
        action="store_true",
        default=True,
        help="Generate and export spatial density heatmaps.",
    )
    parser.add_argument(
        "--heatmap-overlay",
        type=str,
        default="final_heatmap_overlay.jpg",
        help="Output path for the blended spatial heatmap image.",
    )
    parser.add_argument(
        "--heatmap-pure",
        type=str,
        default="final_heatmap_pure.jpg",
        help="Output path for the pure density heatmap image.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights file or name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cpu', 'cuda', '0', etc.",
    )
    parser.add_argument(
        "--line-in",
        type=float,
        default=0.40,
        help="Fractional y-position of the upper IN line (0.0 to 1.0).",
    )
    parser.add_argument(
        "--line-out",
        type=float,
        default=0.75,
        help="Fractional y-position of the lower OUT line (0.0 to 1.0).",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=20.0,
        help="Gaussian blur sigma for density heatmap smoothing.",
    )
    parser.add_argument(
        "--circle-radius",
        type=int,
        default=15,
        help="Pixel radius stamped per detected person per frame.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.50,
        help="Blending transparency for heatmap overlay on reference frame.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip encoding and saving the annotated video stream.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    pipeline = PeopleFlowPipeline(
        weights=args.weights,
        conf=args.conf,
        line_in_ratio=args.line_in,
        line_out_ratio=args.line_out,
        blur_sigma=args.blur_sigma,
        stamp_radius=args.circle_radius,
        overlay_alpha=args.overlay_alpha,
        device=args.device,
    )
    pipeline.run(
        source=args.source,
        output_path=args.output,
        save_heatmap=args.save_heatmap,
        heatmap_overlay_path=args.heatmap_overlay,
        heatmap_pure_path=args.heatmap_pure,
        write_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
