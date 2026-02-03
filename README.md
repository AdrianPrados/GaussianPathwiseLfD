# **Task Generalization with Pathwise Conditioning of Gaussian Process for Learning from Demonstration**

<p align="center">
  <img src="docs/images/EsquemaGeneral.jpg" height=200 />
</p>

To effectively operate in human-centered environments, robots must possess the capability to rapidly adapt to novel and changing situations. Techniques such as Learning from Demonstration enable fast learning without the need for explicit coding. However, in certain cases they exhibit limitations in generalizing beyond the set of demonstrations, which constrains their ability to rapidly adapt to unforeseen scenarios. In this work, we present a movement primitive learning algorithm based on Gaussian Processes, combined with a zero-shot adaptation to new via-points without requiring retraining, through Pathwise Conditioning. The algorithm not only learns the movement policy but is also capable of adapting it rapidly while preserving prior knowledge. The method has been evaluated through comparisons against other state-of-the-art approaches, experiments in simulated environments, as well as on a real robotic platform, generating new solutions for learned tasks by modifying via-points in both position and orientation.


# Installation
To be used on your device, follow the installation steps below.


## Install miniconda (highly-recommended)
It is highly recommended to install all the dependencies on a new virtual environment. For more information check the conda documentation for [installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and [environment management](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). For creating the environment use the following commands on the terminal.

```bash
conda create -n PathwiseGP python=3.10.18
conda activate PathwiseGP
```

### Install repository
Clone the repository in your system.
```bash
git clone https://github.com/AdrianPrados/GaussianPathwiseLfD.git
```
# **Thrird-party dependencies**
To completely use our code, you need to take a look to this other codes :
- [`ADAMSim`](https://github.com/Mobile-Robots-Group-UC3M/AdamSim): Simulator for the ADAM bimanipulator robot in PyBullet. Exact model of the real robot ADAM. This repository contains the URDF files and the necessary code to simulate the robot in PyBullet.
- [`ArucoDetection`](https://github.com/Mobile-Robots-Group-UC3M/ArucoDetection): Code for the detection of ArUco markers using OpenCV. This repository contains the necessary code to detect ArUco markers in images and estimate their pose. This repo contains a small code to transform the marker to PyBullet coordinates.
- [`Pathwise Conditioning`](https://github.com/j-wilson/GPflowSampling): Implementation of Pathwise Conditioning for Gaussian Processes using GPflow. This repository contains the necessary code to perform Pathwise Conditioning on Gaussian Processes using the GPflow library.

# **Algorithm execution**
In this repo we present a series of examples to show the functionality of the algorithm. The main files are:
- [`PathwiseGMP_2D.py`](./PathwiseGMP_2D.py): Example of the 2D functionality of the Pathwise Gaussian Movement Primitives. A set of demonstrations is provided, and the algorithm learns the movement and adapts it to new via-points.
- [`PathwiseGMP_3D.py`](./PathwiseGMP_3D.py): Example of the 3D functionality of the Pathwise Gaussian Movement Primitives. A set of demonstrations is provided, and the algorithm learns the movement and adapts it to new via-points.

- [`OrientationExample2D.py`](./OrientationExample2D.py): Example of the method applied to orientation learning in 2D. A set of demonstrations is provided, and the algorithm learns the orientation and position changes and adapts it to new via-points.

- [`OrientationExample3D.py`](./OrientationExample3D.py): Example of the method applied to orientation learning in 3D. A set of demonstrations is provided, and the algorithm learns the orientation and position changes and adapts it to new via-points.

- [`InteractiveExample2D.py`](./InteractiveExample2D.py): Interactive example in 2D to see how the methods works in real time. You can move the via-points with the mouse and see how the trajectory is adapted in real time.

- [`InteractiveExample3D.py`](./InteractiveExample3D.py): Interactive example in 3D to see how the methods works in real time. You can move the via-points with the mouse and see how the trajectory is adapted in real time.

- [`kukaInteractive3D.py`](./kukaInteractive3D.py): Interactive mode with IIWA Kuka robotic arm.

- [`ADAMInteractive.py`](./ADAMInteractive.py): Interactive mode with ADAM robot (requires [`ADAMSim`](https://github.com/Mobile-Robots-Group-UC3M/AdamSim)).

- [`kukaAruco.py`](./kukaAruco.py): Interactive mode with IIWA Kuka robotic arm simulated using the Aruco with a real camera.

- [`kukaArucoOrientation.py`](./kukaArucoOrientation.py): Interactive mode with IIWA Kuka robotic arm simulated using the Aruco with a real camera. This controls also the orientation

- [`ArucoADAMOrientation.py`](./ArucoADAMOrientation.py): Interactive mode with ADAM robot simulated using the Aruco with a real camera. This allows the control of the simulated ADAM robot in position and orientation using Aruco markers and also the real robot.

> [!NOTE]  
> For the codes that requires camera to detect the Arucos, we have used a Realsense D435i camera. You can use any other camera that is compatible with OpenCV but the code has to be adapted

> [!NOTE]  
> The code and example is under maintenance and under revision. There can be bugs or problems. Please if you detect one, create a issue and I will try to fix it as soon as posible :construction_worker:.


### **Experiments with robot manipulator**
To test the efficiency of the algorithm, experiments have been carried out with a manipulator in a real environment. For this we have created some paths using our LfD method and after that we have applied somo real-time corrections to the via-points using Aruco markers detected with a camera. The robot has to adapt its trajectory in real-time to reach the new via-points.

<p align="center">
  <img src="docs/images/IntroPaper.jpg" height=300 />
</p>

The video with the solution is provided on [Youtube](https://www.youtube.com/watch?v=TWVs3DqLf6g). We have also created a webpage where you can see in detail the results of the experiments. You can access clicking on [WebPage](https://adrianprados.github.io/GaussianPathwiseLfD/).

# Citation
If you use this code, please quote our work (still not citation) :blush:

And other interesting works related to Pathwise Conditioning:
```bibtex
@inproceedings{wilson2020efficiently,
  title={Efficiently sampling functions from Gaussian process posteriors},
  author={Wilson, James and Borovitskiy, Viacheslav and Terenin, Alexander and Mostowsky, Peter and Deisenroth, Marc},
  booktitle={International Conference on Machine Learning},
  pages={10292--10302},
  year={2020},
  organization={PMLR}
}
```
``` bibtex
@article{wilson2021pathwise,
  title={Pathwise conditioning of Gaussian processes},
  author={Wilson, James T and Borovitskiy, Viacheslav and Terenin, Alexander and Mostowsky, Peter and Deisenroth, Marc Peter},
  journal={Journal of Machine Learning Research},
  volume={22},
  number={105},
  pages={1--47},
  year={2021}
}
```

