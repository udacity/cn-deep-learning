# DeepRL Quadcopter Controller

_Teach a Quadcopter How to Fly!_

In this project, you will design a Deep Reinforcement Learning agent to control several quadcopter flying tasks, including take-off, hover and landing.


# Table of Contents

- [Install](#install)
- [Download](#download)
- [Develop](#develop)
- [Submit](#submit)


# Install

This project uses ROS (Robot Operating System) as the primary communication mechanism between your agent and the simulation. You can either install it on your own machine ("native install"), or use a Udacity-provided Virtual Machine (recommended).

## ROS Virtual Machine

Download the compressed VM disk image and unzip it:

- Compressed VM Disk Image: [RoboVM_V2.1.0.zip](https://s3-us-west-1.amazonaws.com/udacity-robotics/Virtual+Machines/Lubuntu_071917/RoboVM_V2.1.0.zip)
- MD5 checksum: `MD5(Ubuntu 64-bit Robo V2.1.0.ova)= 95bfba89fbdac5f2c0a2be2ae186ddbb`

You will need a Virtual Machine player to run the VM, such as VMWare or VirtualBox:

- [VMWare](http://www.vmware.com/): If you use a Windows/Linux system, you can get [Workstation Player](https://www.vmware.com/products/workstation-player/workstation-player-evaluation.html) for free, or if you're on a Mac, you can get a trial of [Fusion](https://www.vmware.com/products/fusion.html).
- [VirtualBox](https://www.virtualbox.org/): Download and install the appropriate version for your system.  On a Mac, you will most likely need to update security settings to allow for a proper installation, as you may otherwise have issues with the Kernal drivers not being installed.  When installing VitualBox, follow the directions [here](https://apple.stackexchange.com/questions/300510/virtualbox-5-1-8-installation-didnt-install-kernel-extensions-how-do-i-fix-thi), and allow 'Oracle America' to load inside of Preferences > Security & Privacy > General.

Open your VM player, and then "Open" / "Import" the VM disk image that you just unzipped (the `.ova` file).

Configure the settings for your VM to allocate at least 2 processors and 4GB of RAM (more the merrier!). Now launch the VM, and follow the on-screen instructions for one-time setup steps.

- Username: `robond`
- Password: `robo-nd`

To open a terminal in your VM, press `Ctrl+Alt+T`. If prompted "Do you want to source ROS?", answer `y` (yes). This is where you will execute your project code.

## ROS Native Install

If you choose to install ROS (Robot Operating System) on your own machine, it is recommended that you use Ubuntu 16.04 LTS as your operating system. To install ROS, please follow the instructions here: [ROS Installation](http://wiki.ros.org/kinetic/Installation)

_Note: This method is not supported by Udacity. If you have trouble performing a native install of ROS, please visit [ROS answers](http://answers.ros.org/questions/) or you can try troubleshooting your install with other students in the Udacity Robotics Slack community ([robotics.udacity.com](https://www.robotics.udacity.com)) in the **#ros** channel._


# Download

## Project Code

On the machine where you have installed ROS (a VM, or your local machine), create a directory named `catkin_ws`, and inside it create a subdirectory named `src`. If you're using a VM, you can also share a folder on your file-system between the host and VM. That might make it easier for you to prepare your report and submit your project for review.


Now clone this repository or download it inside the `src` directory. This is where you will develop your project code. 

So, to do the above, inside the VM terminal:

```bash
$ cd ~
$ mkdir catkin_ws
$ cd catkin_ws
$ mkdir src
$ cd src
$ git clone https://github.com/udacity/RL-Quadcopter.git
```

Your folder structure should look like the following (ROS has a fairly complicated build system, as you will see!):

```
- ~/catkin_ws/
  - src/
    - RL-Quadcopter/
      - quad_controller_rl/
        - ...
```

The root of this structure (`catkin_ws`) is a [catkin workspace](http://wiki.ros.org/catkin/workspaces), which you can use to organize and work on all your ROS-based projects (the name `catkin_ws` is not mandatory - you can change it to anything you want).

## Python Packages

First, install `pip3`:

```bash
$ sudo apt-get update
$ sudo apt-get -y install python3-pip
```

Next, install the Python packages necessary for this project, listed in `requirements.txt`:

```bash
$ pip3 install -r requirements.txt
```

You may need some additional packages, depending on what framework or library you intend to use, e.g. TensorFlow, Keras, PyTorch, etc. Now is a good time to ensure you have these installed.

## Simulator

Download the Udacity Quadcopter Simulator, nicknamed **DroneSim**, for your host computer OS [here](https://github.com/udacity/RoboND-Controls-Lab/releases). 

To start the simulator, simply run the downloaded executable file. You may need to run the simulator _after_ the `roslaunch` step mentioned below in the Run section, so that it can connect to a running ROS master.

_Note: If you are using a Virtual Machine (VM), you cannot run the simulator inside the VM. You have to download and run the simulator for your **host operating system** and connect it to your VM (see below)._

### Connecting the Simulator to a VM

If you are running ROS in a VM, there are a couple of steps necessary to make sure it can communicate with the simulator running on your host system. If not using a VM, these steps are not needed.

#### Enable Networking on VM

- **VMWare**: The default setting should work. To verify, with the VM runnning, go to the Virtual Machine menu > Network Adapter. NAT should be selected.
- **VirtualBox**:
  1. In the VirtualBox Manager, go to Global Tools (top-right corner, above the penguin) > Host Network Manager.
  2. Create a new Host-only Network. You can leave the default settings, e.g. Name = "vboxnet0", Ipv4 Address/Mask = "192.168.56.1/24", and DHCP Server enabled.
  3. Switch back to Machine Tools, and with your VM selected, open its Settings.
  4. Go to the Network tab, change "Attached to" (network type) to "Host-only Adapter", and pick "vboxnet0" from the "Name" dropdown.
  5. Hit Ok to save, and (re)start the VM.

#### Obtain IP Addresses for Host and VM

In a terminal on your host computer, run `ifconfig`. It will list all the network interfaces available, both physical and virtual. There should be one named something like `vmnet` or `vboxnet`. Note the IP address (`inet` or `inet addr`) mentioned for that interface, e.g. `192.168.56.1`. This is your **Host IP address**.

Do the same inside the VM. Here the interface may have a different name, but the IP address should have a common prefix. Note down the complete IP address, e.g. `192.168.56.101` - this your **VM IP address**.

#### Edit Simulator Settings

Inside the simulator's `_Data` or `/Contents` folder (on Mac, right-click the app > Show Package Contents), edit `ros_settings.txt`:

- Set `vm-ip` to the **VM IP address** and set `vm-override` to `true`.
- Set `host-ip` to the **Host IP address** and set `host-override` to `true`.

The host and/or VM's IP address can change when it is restarted. If you are experiencing connectivity problems, be sure to check that the actual IP addresses match what you have in `ros_settings.txt`.


# Develop

Starter code is provided in `quad_controller_rl/` with all the Python modules (`.py` files) under the `src/quad_controller_rl/` package, and the main project notebook under `notebooks/`. Take a look at the files there, but you do not have to make any changes to the code at this point. Complete the following two steps first (**Build** and **Run**), to ensure your ROS installation is working correctly.

## Build

To prepare your code to run with ROS, you will first need to build it. This compiles and links different modules ("ROS nodes") needed for the project. Fortunately, you should only need to do this once, since changes to Python scripts don't need recompilation.

- Go to your catkin workspace (`catkin_ws/`):

```bash
$ cd ~/catkin_ws/
```

- Build ROS nodes:

```bash
$ catkin_make
```

- Enable command-line tab-completion and some other useful ROS utilities:

```bash
$ source devel/setup.bash
```

## Run

To run your project, start ROS with the `rl_controller.launch` file:

```bash
$ cd ~/catkin_ws/src/RL-Quadcopter/quad_controller_rl/launch
$ roslaunch quad_controller_rl rl_controller.launch
```

You should see a few messages on the terminal as different nodes get spun up. Now you can run the simulator, which is a separate Unity application (note that you must start ROS first, and then run the simulator). Once the simulator initializes itself, you should start seeing additional messages in your ROS terminal, indicating a new episode starting every few seconds. The quadcopter in the simulation should show its blades running as it gets control inputs from the agent, and it should reset at the beginning of each episode.


Tip: By defaults, you will need to follow the steps for build and run each time that you wish to run your quadcopter simulation.  If you get tired of this two-step startup process, edit the `quad_controller_rl/scripts/drone_sim` script and enter a command that runs the simulator application. (An example can be found [here](https://discussions.udacity.com/t/importerror-when-running-roslaunch-quad-controller-rl-rl-controller-launch/569530/2).  It will then be launched automatically with ROS!

_Note: If you want to learn more about how ROS works and how you can use it for robotics applications, you may enroll in Udacity's [Robotics Nanodegree Program](https://www.udacity.com/robotics) and can take the [ROS Essentials](https://classroom.udacity.com/nanodegrees/nd209/parts/af07ae99-7d69-4b45-ab98-3fde8b576a16) module._

## Implement

Once you have made sure ROS and the simulator are running without any errors, and that they can communicate with each other, try modifying the code in `agents/policy_search.py` - this is a sample agent that runs by default (e.g. add a `print` statement). Every time you make a change, you will need to stop the simulator (press `Esc` with the simulator window active), and shutdown ROS (press `Ctrl+C` in the terminal). Save your change, and `roslaunch` again.

Now you should be ready to start coding! Open the project notebook for further instructions (assuming you are in your catkin workspace):

```bash
$ jupyter notebook src/RL-Quadcopter/quad_controller_rl/notebooks/RL-Quadcopter.ipynb
```

# Submit

Complete the required sections in the notebook. Once done, save/export the notebook as a PDF or HTML file. This will serve as your project report.

If you are enrolled in a Udacity Nanodegree program, you can submit your completed project for review. Prepare and upload a zip file that includes the following:

- `RL-Quadcopter.ipynb`: Project notebook with all sections completed.
- `RL-Quadcopter.pdf` / `RL-Quadcopter.html`: PDF or HTML report (which can be an export of the notebook).
- `quad_controller_rl/`: Python package with your code in the following sub-packages:
  - `tasks/`: Your implementation of each task required for this project.
  - `agents/`: Agents you've implemented for the corresponding tasks.
  - Any other supporting code or other files used by your project.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.
