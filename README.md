# Predictive Neural Ray Tracing with Learned Radiance Caching

An experimental research prototype exploring real-time global illumination through the lens of neural rendering and temporal prediction. This project implements a lightweight Neural Radiance Cache (NRC) accelerated by multi-resolution hash grid encoding, featuring a novel "predictive oracle" mechanism to stabilize lighting under rapid camera transitions.

## Overview
Real-time global illumination (GI) remains a computationally expensive challenge in computer graphics, often requiring thousands of rays per pixel to converge. This project bypasses high-sample counts by training a neural network to learn the radiance field of a scene in real-time. By treating GI as a learning problem, we can achieve stable, high-quality lighting at interactive frame rates.

The core innovation is a **Predictive Oracle**, which extrapolates the camera's trajectory to pre-train the network on future states, significantly reducing temporal lag and "ghosting" artifacts during movement.

## Key Ideas

### Neural Radiance Cache (NRC)
Instead of tracing complex light paths to completion, we terminate rays at the first bounce and query a small, high-throughput MLP. The MLP is trained online to predict the indirect radiance at any given point and normal in the scene.

### Hash Grid Encoding
To handle high-frequency spatial details, we utilize a multi-resolution hash grid (inspired by Instant-NGP). This maps 3D world coordinates into a feature-rich latent space, allowing the shallow MLP to represent complex lighting environments with minimal parameters.

### Predictive Oracle Training
The system monitors the camera's linear and angular velocity to extrapolate its position and orientation several frames into the future. By allocating a portion of the training budget to these extrapolated "future views," the model proactively learns the radiance for regions the user is about to see.

## Method
1.  **Ray Generation:** Cast primary rays from the current camera frustum.
2.  **Intersection:** Compute ray-scene intersections (planes and boxes) using optimized slab methods.
3.  **Direct Lighting:** Calculate analytical direct light with shadow-ray verification.
4.  **Indirect Query:** Use the Hash-MLP to predict the indirect component at the hit point.
5.  **Bootstrap Training:** Sample random secondary rays to generate ground-truth radiance targets for online network optimization.
6.  **Oracle Step:** Extrapolate camera state, trace future-view rays, and perform a weighted training pass on these predicted hits.

## Features
*   **Real-time Neural Rendering:** High-performance inference loop using PyTorch.
*   **Learned Global Illumination:** Dynamic indirect lighting that adapts as the network converges.
*   **Temporal Prediction:** Future-aware training loop for motion stability.
*   **Interactive UI:** OpenCV-based loop with real-time "Oracle View" debug overlay.

## Results
The system achieves interactive frame rates on modern GPU hardware. Compared to naive monte-carlo sampling at the same ray budget, the learned radiance cache provides significantly smoother results with lower variance. The predictive training successfully mitigates the "laggy" convergence typical of online-trained radiance caches during fast rotations.

## Code Structure
*   `model.py`: Implements the `HashEmbedder` and the `HashNRC` MLP.
*   `scene.py`: Defines the geometry, light sources, and ray-intersection logic.
*   `renderer.py`: Contains the core `trace_and_shade` logic and the `render_loop` training cycle.
*   `main.py`: Handles the camera physics, OpenCV interaction, and the program entry point.
*   `config.py`: Centralized management of resolution, hardware settings, and hyper-parameters.

## Notes
*   This is an **experimental research prototype** designed for exploring neural rendering concepts.
*   The implementation is focused on clarity and modularity rather than maximum low-level optimization.
*   Developed as a testbed for temporal prediction in radiance caching.
