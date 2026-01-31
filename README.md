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
conda create -n GBPDualArm python=3.10.16
conda activate GBPDualArm
```

### Install repository
Clone the repository in your system.
```bash
git clone https://github.com/AdrianPrados/GaussianBeliefPropagationDualArm.git
```
# **Code Organization**
The code is organized as follows:
- [`factor_graphs`](/factor_graphs/): It contains the necessary files for the generation of communication graphs. These implementations establish the creation of the graphs, the generation of factor graphs, and the creation of Gaussian variables for the communication of Gaussian Belief Propagation.
- [`motionGBP`](/motionGBP/): They contain the generation of factors, nodes, and obstacles for GBP. The files are divided into 2D and 3D.
- [`Adam_sim`](/Adam_sim/): It contains the models of the ADAM bimanipulator robot, which can be used for simulations in PyBullet.
> [!CAUTION]  
> The simulator for Adam is not fully completed, we have just uploaded in this repository a beta version.

> [!NOTE]  
> The code for the Learning from Demosntration part is not included in this repository. If you are interested in this part, please visit this repository: [TP-GMM with few demonstrations](https://github.com/AdrianPrados/Learning-and-generalization-of-task-parameterized-skills-through-few-human-demonstrations)
# **Algorithm execution**
There are different codes that you can try with our implementation:
- [`test_factorgraph.py`](./test_factorgraph.py): Provides the definition of the Gaussian Process class generated for the demonstration learning process.
- [`test_motion2D.py`](./test_motion2D.py): Example of the 2D functionality of two spherical agents that move freely in space. Obstacles and additional agents can be added to test the effectiveness of the algorithm.

- [`test_motion3D.py`](./test_motion3D.py): Example of the 3D functionality of two spherical agents that move freely in space. Obstacles and additional agents can be added to test the effectiveness of the algorithm.

- [`2armsBullet.py`](./2armsBullet.py): Example of functionality with two opposing IIWA robotic arms. In these cases, static obstacles can be added. The dynamic obstacles will be the robot arms themselves.

- [`AdamGBP.py`](./AdamGBP.py): Example of use with the ADAM robot. In this case, reference paths are provided, which have been learned independently for each arm using LfD. The algorithm follows them and makes corrections when encountering another agent or an obstacle in the environment.


### **Experiments with robot manipulator**
To test the efficiency of the algorithm, experiments have been carried out with a manipulator in a real environment. For this purpose, a series of data have been taken by means of a kinesthetic demonstration and then tested using the method t avoid the collision between both of the arms.

<p align="center">
  <img src="docs/static/images/IntroPaper.jpg" height=300 />
</p>

The video with the solution is provided on [Youtube](https://youtu.be/4wB8RAer-kw). We have also created a webpage where you can see in detail the results of the experiments. You can access clicking on [WebPage](https://adrianprados.github.io/GaussianBeliefPropagationDualArm/).

# Citation
If you use this code, please quote our works :blush:

```bibtex
@inproceedings{prados2025coordination,
  title={Coordination of Learned Decoupled Dual-Arm Tasks through Gaussian Belief Propagation},
  author={Prados, Adrian and Espinoza, Gonzalo and Moreno, Luis and Barber, Ramon},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={15917--15924},
  year={2025},
  organization={IEEE}
}
```