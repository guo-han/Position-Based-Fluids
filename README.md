<p align="center">
  <h1 align="center">Bunny in the Water</h1>
  <p align="center">
    <a href="https://github.com/DecAd3"><strong>Longteng Duan*</strong></a>
    ·
    <a href="https://github.com/guo-han"><strong>Guo Han*</strong></a>
    ·
    <a href="https://github.com/Ribosome-rbx"><strong>Boxiang Rong*</strong></a>
  </p>
  <p align="center"><strong>(* Equal Contribution)</strong></p>
  <h3 align="center"> <a href="">Slides</a> | <a href="">Presentation</a> | <a href="">Demo</a> </h3>
  <div align="center"></div>
</p>

[![](./imgs/all_high_res.png)](https://www.youtube.com/watch?v=6cz7K6m6m8M)
<p align="center">
    (Click to View our Demo Video)
</p>
<p align="center">
    Here we present our group project developed for the <a href = "https://crl.ethz.ch/teaching/PBS23/index.html">Physically-based Simulation in Computer Graphics</a> course at ETH Zurich.  Our work involved the implementation of several key features, including position-based fluids (PBF), handling of simple static rigid body, and the generation of diffuse materials such as spray, foam, and bubbles.
</p>

## Presentation Video

## Environment Setup

## Code Usage
### Real-time Simulation
The real-time simulation provides a Taichi GGUI interface for visualization. You'll witness a dynamic scene featuring a board in motion, creating ripples as it sways back and forth while various diffuse particles (green: spray; white: foam; red: bubble) are dynamically generated. Amidst this aquatic environment, a stationary bunny stands amidst the flowing water.

The simulation process can run at 25+FPS on RTX 3070 GPU.

If your computer does not have a GPU, please change line 15 of `main.py` to `ti.init()`
```
python main.py
```
### Rendering
To prepare for rendering, follow these steps within the `main.py` file initially:

1. Set `bake_mesh = True` on line 132.
2. Specify the starting and ending frames for mesh creation on line 140.
3. Execute `main.py` until the output confirms the completion of mesh baking.

Afterward, execute `python blender_render.py` to commence image rendering. Please note that rendering the entire video sequence may require a considerable amount of time.