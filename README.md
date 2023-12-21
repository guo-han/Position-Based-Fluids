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
    Here we present our group project developed for the <a href = "https://crl.ethz.ch/teaching/PBS23/index.html">Physically-based Simulation in Computer Graphics</a> course at ETH Zurich.  Our work involved the implementation of several key features, including position-based fluids (PBF), handling of simple static rigid body, and generating diffuse materials such as spray, foam, and bubbles.
</p>

## Presentation Video

## Environment Setup
To run the real-time simulation, you need to install <a href="https://www.taichi-lang.org/">Taichi</a> and <a href="http://www.open3d.org/">Open3d</a>. The Python version we use is 3.10.13, the Taichi version is taichi==1.6.0, and the Open3d version is open3d==0.17.0.

To run the rendering, you need to install <a href="https://github.com/InteractiveComputerGraphics/splashsurf">splashsurf</a> and <a href="https://www.blender.org/">Blender python API </a>
## Code Usage
### Real-time Simulation
In this project, we built from scratch a Real-time Position-based Fluid Simulation pipeline, with advanced features like: static rigid body collision handling and diffuse particle generation (foam, spray, and bubbles). It is fully developed with the Taichi language and provides a GGUI interface for visualization. We also provide scripts to export the particle-based simulation results into mesh, and render the final scene with Blender Python APIs.

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

## Reference
[1] Macklin, Miles, and Matthias Müller. "Position based fluids." ACM Transactions on Graphics (TOG) 32.4 (2013): 1-12.

[2] Ihmsen, Markus, et al. "Unified spray, foam and air bubbles for particle-based fluids." The Visual Computer 28 (2012): 669-677.

[3] J. Bender, D. Koschier, T. Kugelstadt and M. Weiler, "Turbulent Micropolar SPH Fluids with Foam," in IEEE Transactions on Visualization and Computer Graphics, vol. 25, no. 6, pp. 2284-2295, 1 June 2019, doi: 10.1109/TVCG.2018.2832080.

[4] Taichi Blog for Collision Handling: https://docs.taichi-lang.org/blog/acclerate-collision-detection-with-taichi

[5] Taichi PBF 2D Example by Ye Kuang: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/pbf2d.py 

[6] SPlisHSPlasH Library for Diffuse Particles Synthesis: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH 
