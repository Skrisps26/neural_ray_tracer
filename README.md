# Predictive Neural Radiance Caching

### TL;DR
- Reframes global illumination as a regression problem using an MLP to predict indirect lighting.
- Achieves real-time performance (~60 FPS) with dynamic scenes and camera movement.
- Predictive "oracle" training improves stability by learning from extrapolated future camera states.

### Demo

![Current Frame (Predicted Lighting)](Untitled2.jpg)
*Current Frame (Predicted Lighting)*

![Oracle / Future Prediction (Lookahead Training Signal)](Untitled.jpg)
*Oracle / Future Prediction (Lookahead Training Signal)*

### Key Idea
The system treats global illumination as a spatial regression problem. Instead of performing expensive multi-bounce ray tracing per pixel, a shallow MLP accelerated by multi-resolution hash grid encoding learns to predict the indirect radiance field. By integrating a predictive oracle that extrapolates camera motion, the network pre-trains on upcoming views, effectively mitigating the temporal lag typically seen in online-trained radiance caches.

### Uncertainty-Aware Prediction
The model is extended to predict both radiance and an uncertainty estimate for each sample. Instead of regressing only RGB values, the network outputs RGB radiance and a log-variance term representing prediction uncertainty.

Training uses a Gaussian negative log-likelihood objective, allowing the model to express confidence in its predictions. This enables:
- **Reduced penalty for ambiguous regions** (e.g., shadow boundaries).
- **Enhanced stability** under rapid motion and dynamic lighting.
- **Foundation for adaptive rendering** (potential for falling back to high-fidelity ray tracing in high-uncertainty regions).

This is a lightweight approximation of confidence-aware neural rendering integrated directly into the real-time training loop.

### Relation to Prior Work
The approximation of global illumination using neural networks is an established direction in computer graphics. Prior work, such as Neural Radiance Caching (NRC) and related neural rendering approaches, has demonstrated the viability of training neural networks online to predict indirect lighting during rendering. These systems typically rely on specialized RTX pipelines and high-end infrastructure to maintain performance.

This project is a systems-level experiment designed for simplicity and accessibility. Unlike production-grade implementations, this system is intentionally lightweight and built to run on consumer hardware without specialized ray-tracing cores. It frames radiance estimation as a straightforward regression problem using an MLP, avoiding complex rendering pipelines. The primary technical distinction is the use of a predictive "oracle" training signal: by extrapolating future camera states based on motion, the system leverages temporal coherence to guide learning, rather than relying purely on the stochastic sampling methods used in traditional radiance caching.

### Results
- **Performance:** Reduced average frame time from ~26ms to ~11ms (~2.3x speedup).
- **Framerate:** Stable ~60 FPS on GPU.
- **Stability:** High temporal coherence during rapid camera translation and rotation.

### How it Works
1.  **Direct Rendering:** Execute primary ray tracing and compute analytical direct lighting.
2.  **Neural Inference:** Query the Hash-MLP for the indirect radiance component and uncertainty estimate at primary hit locations.
3.  **Bootstrap Training:** Sample random secondary rays to generate ground-truth radiance targets for online optimization.
4.  **Predictive Oracle:** Extrapolate camera trajectory to generate future-view rays, populating the cache before the viewport arrives.

### Running the Project
```bash
pip install -r requirements.txt
python main.py
```
*Controls: WASD for movement, Q/E for rotation, ESC to quit.*

### Key Insight
Learning-based rendering systems benefit from modeling uncertainty explicitly — not all regions of a scene are equally predictable. By allowing the model to express confidence, we can move toward hybrid systems that combine fast neural approximations with selective high-quality computation. A predictive training loop remains the primary mechanism for maintaining visual stability under motion.

This project should be viewed as a practical exploration of neural rendering ideas under constrained settings, rather than a replacement for production-grade or research-grade global illumination systems. It serves as a testbed for investigating the relationship between temporal prediction and online network convergence.
