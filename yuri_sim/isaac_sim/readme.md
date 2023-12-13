# CuRobo demo
Nvidia Isaac Sim and CuRobo integration demo, the arm will follow the target cube and avoid obstacles.
Default robot is Franka Panda, and it supports command line parameters.
```bash
omni_python yuri_arm/yuri_sim/isaac_sim/curobo/motion_generation.py
```
Notes:
`omni_python` means `python.sh` under the isaac sim directory, you can add alias to ~/.bashrc:
```bash
echo "alias omni_python='~/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/python.sh'" >> ~/.bashrc
```
