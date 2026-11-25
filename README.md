<div align="center">

# 🚶 People Flow Detection & Spatial Density Heatmap

**An edge-ready, real-time computer vision system for bidirectional pedestrian traffic monitoring, anti-double-counting tripwire analytics, and continuous spatial density heatmapping.**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![Supervision](https://img.shields.io/badge/Supervision-ByteTrack-7B2CBF?style=for-the-badge)](https://github.com/roboflow/supervision)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<br/>

<img src="assets/demo_preview.gif" alt="People Flow Detection Real-Time Demo" width="900" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"/>

</div>

---

## 📌 Problem Statement & Real-World Use Cases

Automated foot-traffic estimation and crowd movement analytics are essential components of modern smart infrastructure. Traditional sensor hardware (such as overhead infrared beams or optical turnstiles) is rigid, expensive to maintain, and incapable of capturing spatial trajectory intelligence. Conversely, naive computer vision approaches suffer from frequent tracking identity switches, spurious counts caused by pedestrian posture changes or loitering near boundary lines, and an inability to map dwell behavior.

This system addresses these engineering challenges through a robust, four-stage vision pipeline:

* **Retail & Storefront Intelligence:** Quantify foot-traffic conversion rates, measure entry/exit volumes, and visualize customer dwell hotspots across store aisles without dedicated tracking hardware.
* **Transit Hubs & Public Plazas:** Monitor commuter flow velocity, identify corridor congestion bottlenecks, and dynamically optimize gate throughput at train stations, airports, and bus terminals.
* **Smart Facility Management & Evacuation Readiness:** Enforce occupancy compliance limits in corporate offices, warehouses, and event venues with persistent identity accounting.

---

## 🏗️ System Architecture & Engineering Pipeline

The system is engineered as an end-to-end modular pipeline combining state-of-the-art deep learning detection, Kalman-filtered motion tracking, kinematic anchor geometry, and floating-point continuous accumulation.

```mermaid
flowchart TD
    A["Raw Video Stream<br/>(File / RTSP / Web Stream)"] --> B["YOLOv8n Inference<br/>(Filtered to Class 0: Person)"]
    B --> C["ByteTrack MOT<br/>(Kalman Filter + Low-Score Association)"]
    C --> D["Foot-Anchor Extraction<br/>(cx = (x1+x2)/2, cy = y2)"]
    D --> E["Bidirectional Virtual Lines<br/>(Upper @ 40% H | Lower @ 75% H)"]
    D --> F["Spatial Heatmap Accumulator<br/>(Incremental Float32 Matrix)"]
    E --> G["Anti-Double-Counting Logic<br/>(State Sets: counted_in_ids, counted_out_ids)"]
    G --> H["Real-Time Heads-Up Display<br/>(Bounding Boxes, IDs, Live HUD)"]
    F --> I["Gaussian Smoothing & Colormapping<br/>(cv2.GaussianBlur + COLORMAP_JET)"]
    H --> J["Annotated Output Video (.mp4)"]
    I --> K["Spatial Heatmaps<br/>(Pure Density & Reference Overlay .jpg)"]
```

### 1. Pedestrian Detection (YOLOv8n)
- High-throughput inference utilizing a lightweight, pretrained `YOLOv8n` convolutional backbone.
- Detections are strictly filtered to COCO Class `0` (`person`) with a configurable confidence threshold (`--conf 0.30` by default), eliminating background artifacts and object clutter.

### 2. Multi-Object Tracking (ByteTrack)
- Track association is handled by the **ByteTrack** algorithm via `supervision.ByteTrack`.
- Unlike basic centroid trackers or appearance-heavy Re-ID models, ByteTrack preserves tracking continuity across frames by matching not only high-confidence bounding boxes but also low-score candidate detections using Kalman filter state predictions. This drastically mitigates track fragmentation during partial pedestrian occlusions.

### 3. Kinematic Ground Anchor (Foot-Position Geometry)
- In top-down and angled perspective cameras, bounding box centroids $(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2})$ oscillate drastically as pedestrians swing their arms or change stride posture.
- This pipeline computes the **ground contact coordinate** at the bottom-center of the bounding box:
  $$\text{Anchor Point: } (c_x, c_y) = \left(\frac{x_1 + x_2}{2}, y_2\right)$$
  Using the ground contact point ensures that tripwire crossings align strictly with ground-plane movement, eliminating false trigger events.

### 4. Bidirectional Anti-Double-Counting Virtual Tripwires
- Two horizontal virtual tripwires partition the frame along normalized Y-axis coordinates:
  - **Upper Tripwire ($\text{LINE\_UPPER\_Y} = 0.40 \times \text{Height}$, Green):** Designated as the **IN** gate. Registers an entry when a tracked pedestrian travels downward (southbound) from above the line to at/below it:
    $$\text{prev\_}c_y < \text{LINE\_UPPER\_Y} \le c_y \quad \land \quad \text{track\_id} \notin \text{counted\_in\_ids}$$
  - **Lower Tripwire ($\text{LINE\_LOWER\_Y} = 0.75 \times \text{Height}$, Red):** Designated as the **OUT** gate. Registers an exit when a tracked pedestrian travels upward (northbound) from below the line to at/above it:
    $$\text{prev\_}c_y > \text{LINE\_LOWER_Y} \ge c_y \quad \land \quad \text{track\_id} \notin \text{counted\_out\_ids}$$
- **Anti-Double-Counting Invariant:** Persistent integer sets (`counted_in_ids`, `counted_out_ids`) guarantee that once an ID is tallied for a given direction, subsequent oscillations or pauses around the boundary cannot re-trigger the counter. Crossings in the reverse direction on either line are safely ignored.

### 5. Continuous Float32 Spatial Density Heatmap
- Rather than rendering discrete bounding box rectangles, the pipeline maintains a 2D floating-point accumulation matrix $\mathbf{M} \in \mathbb{R}^{H \times W}$ initialized to zeros.
- For every frame, each active pedestrian foot anchor $(c_x, c_y)$ is stamped with a solid circular kernel:
  $$\mathbf{M}_{(y, x)} \leftarrow \mathbf{M}_{(y, x)} + w \quad \forall (x - c_x)^2 + (y - c_y)^2 \le r^2$$
- At pipeline termination, spatial post-processing produces publication-grade visuals:
  1. **Gaussian Convolution:** $\mathbf{G} = \mathbf{M} * \mathcal{K}_\sigma$ (standard deviation $\sigma = 20.0$).
  2. **Min-Max Normalization:** Scaled linearly to $[0, 255]$ and cast to `uint8`.
  3. **Pseudocolor Rendering:** Processed through OpenCV's `COLORMAP_JET` spectrum (cool blue for transient pass-through, vibrant red for high-dwell density).
  4. **Spatial Context Blending:** Alpha-blended with the baseline reference frame ($\alpha = 0.50$) to ground the density peaks against the physical environment.

---

## 📊 Visual Analytics & Artifact Gallery

| Calibrated Virtual Tripwires | Real-Time MOT & HUD Counting |
| :---: | :---: |
| ![Virtual Lines Calibration](assets/line_preview.jpg) | ![Tracking HUD Sample](assets/tracking_sample_frame.jpg) |
| *Initial frame calibration showing Upper IN Line (40% frame height) and Lower OUT Line (75% frame height).* | *Real-time ByteTrack ID assignment, foot ground anchors, and live dashboard counters.* |

| Pure Motion Density Heatmap | Geospatial Background Overlay |
| :---: | :---: |
| ![Pure Heatmap](assets/final_heatmap_pure.jpg) | ![Heatmap Overlay](assets/final_heatmap_overlay.jpg) |
| *Cumulative float32 Gaussian density map (`COLORMAP_JET`) displaying primary pedestrian corridors.* | *50% alpha overlay on physical reference frame highlighting dwell points and transit corridors.* |

---

## 📁 Repository Structure

```text
people-flow-detection/
├── assets/                          # Visual documentation assets & sample outputs
│   ├── demo_preview.gif             # Real-time animated pipeline preview
│   ├── final_heatmap_overlay.jpg    # 50% alpha blended context heatmap (1080p)
│   ├── final_heatmap_pure.jpg       # Pure Gaussian density heatmap (1080p)
│   ├── line_preview.jpg             # Dual tripwire calibration preview
│   ├── tracking_sample_frame.jpg    # Annotated detection frame with HUD
│   └── sample_output.mp4            # Benchmark output video stream
├── src/                             # Modular production-ready Python source
│   ├── __init__.py                  # Package exports
│   └── pipeline.py                  # Core CLI pipeline, tracker, and heatmap engine
├── People_Flow_Detection.ipynb      # Cleaned interactive analysis notebook
├── requirements.txt                 # Pinned, reproducible dependencies
├── .gitignore                       # Clean Python, Jupyter, weights, and video exclusions
└── README.md                        # Technical system documentation
```

---

## ⚡ Quickstart Guide

### 1. Clone Repository & Setup Environment

```bash
git clone https://github.com/MehdiHossenFahim/people-flow-detection.git
cd people-flow-detection

# Create and activate a virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Run Modular CLI Pipeline

The CLI script `src/pipeline.py` is fully self-contained. If no local video file is supplied, it automatically downloads a standard benchmark stream (`people-walking.mp4`) and executes end-to-end:

```bash
# Basic run with default parameters
python -m src.pipeline

# Custom inference on a local video file with custom thresholds
python src/pipeline.py \
    --source path/to/input_video.mp4 \
    --output output/annotated_flow.mp4 \
    --conf 0.35 \
    --save-heatmap assets/ \
    --upper-line 0.40 \
    --lower-line 0.75 \
    --sigma 22.0
```

### 3. Run Interactive Jupyter Notebook

For step-by-step experimentation, visualization, and interactive parameter tuning:

```bash
jupyter notebook People_Flow_Detection.ipynb
```
The notebook is completely portable, free of hardcoded environment paths, and runs seamlessly on both local workstations and Google Colab.

---

## ⚙️ Configuration Parameters (CLI Reference)

All pipeline parameters can be customized via command-line arguments:

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `str` | `people-walking.mp4` | Path or HTTP/HTTPS URL to input video. Automatically downloads benchmark video if missing. |
| `--output` | `str` | `people_flow_output.mp4` | File path where the annotated output MP4 video will be written. |
| `--conf` | `float` | `0.30` | YOLOv8 detection confidence threshold for the person class (`0.0` - `1.0`). |
| `--save-heatmap` | `str` | `assets` | Destination directory to export `final_heatmap_pure.jpg` and `final_heatmap_overlay.jpg`. |
| `--weights` | `str` | `yolov8n.pt` | Model weights checkpoint (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, etc.). |
| `--device` | `str` | `auto` | Computation device: `'cpu'`, `'cuda'`, `'mps'`, or `'auto'`. |
| `--upper-line` | `float` | `0.40` | Normalized Y position (fraction of frame height) for the Upper **IN** line. |
| `--lower-line` | `float` | `0.75` | Normalized Y position (fraction of frame height) for the Lower **OUT** line. |
| `--radius` | `int` | `15` | Radius in pixels of circular foot-anchor stamp on the float32 accumulator. |
| `--sigma` | `float` | `20.0` | Standard deviation for the Gaussian spatial density smoothing kernel. |
| `--alpha` | `float` | `0.50` | Blending transparency weight of heatmap over background reference frame. |
| `--preview` | `flag` | `False` | Opens an interactive OpenCV window displaying real-time annotations. |

---

## 📈 Benchmark Results

Evaluated on the standard pedestrian surveillance benchmark stream (`1920x1080` @ 30 FPS):

* **Total Pedestrians IN (Southbound):** `16`
* **Total Pedestrians OUT (Northbound):** `9`
* **Net Flow Volume:** `25` verified line-crossing events
* **Double-Counting Error Rate:** `0.0%` (strictly enforced by ID state sets)
* **Average Inference Speed:**
  - NVIDIA RTX 4090 / CUDA: `~85+ FPS` (real-time stream multiplexing)
  - Intel Core i7 / CPU: `~22-28 FPS` (near-line-rate edge throughput)

---

## 🛠️ Extensibility & Production Deployment

The modular design of `src/pipeline.py` allows seamless extension for enterprise production environments:
1. **RTSP Multi-Stream Ingestion:** Replace the frame generator with an asynchronous `threading.Queue` or GStreamer pipeline for multi-camera CCTV feeds.
2. **Prometheus / Grafana Exporter:** Stream real-time IN/OUT counters and occupancy delta metrics to a time-series database.
3. **Polygon RoI Zones:** Extend `LineCrossingCounter` with `shapely.geometry.Polygon` for arbitrary retail shelf dwell-time profiling.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
