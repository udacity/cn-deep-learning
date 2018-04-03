# DeepRL 四轴飞行器控制器

_指导四轴飞行器学会飞行！_

在本次项目中，你将设计一个深度强化学习智能体，来控制几个四轴飞行器的飞行任务，包括起飞、盘旋和着陆。


# 目录

- [安装](#install)
- [下载](#download)
- [开发](#develop)
- [提交](#submit)


# 安装

本项目使用 ROS（机器人操作系统）作为你的智能体和模拟之间的主要沟通机制。你可以在你的电脑本地安装 ROS，或是使用优达学城提供的虚拟机（推荐）。

## ROS 虚拟机

下载压缩的 VM 磁盘镜像并解压：

- 压缩的 VM 磁盘镜像：[RoboVM_V2.1.0.zip](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DQN/RoboVM_V2.1.0.zip)
- MD5 校验和：`MD5(Ubuntu 64-bit Robo V2.1.0.ova)= 95bfba89fbdac5f2c0a2be2ae186ddbb`

你还需要一个虚拟机软件来运行 VM，比如 VMWare 或 VirtualBox：

- [VMWare](http://www.vmware.com/)：如果你使用 Windows/Linux 系统，可以免费下载  [Workstation Player](https://www.vmware.com/products/workstation-player/workstation-player-evaluation.html)，如果你使用的是 Mac，可以下载 [Fusion](https://www.vmware.com/products/fusion.html) 的试用版。
- [VirtualBox](https://www.virtualbox.org/)：请下载并安装与你的系统适配的版本。在 Mac 上，你需要更新安全设置来允许软件安装，否则你将由于核心驱动问题而安装失败。在安装 VitualBox 时，请参照 [这里](https://apple.stackexchange.com/questions/300510/virtualbox-5-1-8-installation-didnt-install-kernel-extensions-how-do-i-fix-thi) 的说明，并在偏好设置 > 安全性与隐私 > 通用允许 'Oracle America' 下载。

打开你的 VM 运行软件，接着“打开”/“导入”你刚刚解压的 VM 磁盘镜像（`.ova` 文件）。

请配置你的虚拟机，并分配至少 2 个处理器和 4GB RAM 内存（越多越好！）。现在运行 VM，并根据屏幕上的指示进行配置。

- 用户名：`robond`
- 密码：`robo-nd`

要想在 VM 中打开终端，请按下 `Ctrl+Alt+T`。如果系统提示“Do you want to source ROS?”，选择 `y` （是）。你将在这里执行项目代码。

## ROS 本地安装

如果你选择在你的电脑本地安装 ROS，我们建议你使用 Ubuntu 16.04 LTS 作为操作系统。请参照 [ROS 安装指南](http://wiki.ros.org/kinetic/Installation)进行安装。

_请注意：优达学城并不支持此方法。如果你在本地安装 ROS 时出现问题，请访问 [ROS 疑难解答](http://answers.ros.org/questions/)或优达学城机器人开发的 Slack 社区（[robotics.udacity.com](https://www.robotics.udacity.com)），并在 **#ros** 频道中与其他学员讨论解决方法。_


# 下载

## 项目代码

在安装了 ROS （虚拟机或本地计算机）的机器中创建名为 `catkin_ws` 的目录，并在该目录下创建名为 `src` 的子目录。如果你使用的是虚拟机，你也可以在主机和虚拟机的文件系统里共享文件夹。使用这种方法，你可以更方便地撰写报告和提交项目以供审阅。


现在请将这个代码库克隆或下载至 `src` 目录下。你将在这里开发你的项目代码。

请在终端内输入：

```bash
$ cd ~
$ mkdir catkin_ws
$ cd catkin_ws
$ mkdir src
$ cd src
$ git clone https://github.com/udacity/RL-Quadcopter.git
```

你的目录结构应如下所示（ROS 的编译系统相当复杂，在之后的内容中你也将了解到这一点）：

```
- ~/catkin_ws/
  - src/
    - RL-Quadcopter/
      - quad_controller_rl/
        - ...
```

该结构的根目录（`catkin_ws`）是一个 [catkin workspace](http://wiki.ros.org/catkin/workspaces)，你可以使用它来管理和进行所有基于 ROS 的项目（文件夹名 `catkin_ws` 并非强制，你可以随意更改）。

## Python 包

首先，安装 `pip3`：

```bash
$ sudo apt-get update
$ sudo apt-get -y install python3-pip
```

然后，安装本项目所需的 Python 包，已在 `requirements.txt` 中列出：

```bash
$ pip3 install -r requirements.txt
```

根据你所使用的框架和库，你也许还需要一些额外的包，比如 TensorFlow，Keras，PyTorch 等等。现在请确保你已经安装了这些包。

## 模拟器

为你的主机 OS 在[这里](https://github.com/udacity/RoboND-Controls-Lab/releases)下载优达学城四轴飞行器模拟器，它的昵称为 **DroneSim**。

要想打开模拟器，你只需运行下载的可执行文件即可。你也许需要在运行部分介绍的 `roslaunch` 步骤_之后_打开模拟器，便于将它连接至正在运行的 ROS master。

_请注意：如果你使用的是虚拟机（VM），你无法在 VM 内运行模拟器。你需要在**主机操作系统**中下载并运行模拟器，再将它连接至虚拟机（见下文）_

### 将模拟器连接至 VM

如果你在虚拟机内运行 ROS，你需要通过几个步骤来保证它能与主机系统中的模拟器连接。如果你没有使用虚拟机，可以忽略这些步骤。

#### 在 VM 中允许网络连接

- **VMWare**：使用默认设定即可。为了验证，你可以在运行虚拟机的情况下，打开虚拟机的菜单 > 网络适配器。NAT 一栏应被勾选。
- **VirtualBox**：
  1. 在 VirtualBox Manager 中，打开 Global Tools（右上角，penguin 上方）> Host Network Manager。
  2. 创建一个新的主机模式网络。你可以使用默认配置，比如 Name = "vboxnet0"，Ipv4 Address/Mask = "192.168.56.1/24"，并允许 DHCP 服务器。
  3. 切换回 Machine Tools，选择虚拟机并打开它的设定。
  4. 打开 Network 标签页，将 "Attached to"（网络类型）改成 "Host-only Adapter"，并选择 "Name" 下的 "vboxnet0"。
  5. 点击 Ok 进行保存，并启动（重启）虚拟机。

#### 为主机和虚拟机获取 IP 地址

在主机终端中运行 `ifconfig`。这将显示所有可用的网络接口，包括物理接口和虚拟接口。其中应有名为 `vmnet` 或 `vboxnet` 的接口。请记下该接口的 IP 地址（`inet` 或 `inet addr`），比如 `192.168.56.1`，这是你的**主机 IP 地址**。

请在虚拟机内重复该步骤。在这里接口名称也许有所不同，但 IP 地址的前缀相同。请记下完整的 IP 地址，比如 `192.168.56.101`，这是你的**虚拟机 IP 地址**。

#### 编辑模拟器设定

在模拟器的 `_Data` 或 `/Contents` 文件夹内（在 Mac中请右键点击 app > 显示包目录），编辑 `ros_settings.txt`：

- 将 `vm-ip` 设置为 **虚拟机 IP 地址**，并将 `vm-override` 设置为 `true`。
- 将`host-ip` 设置为 **主机 IP 地址**，并将 `host-override` 设置为 `true`。

主机和/或虚拟机的 IP 地址可以在重启时改变。如果你遇到任何连接问题，请确保实际的 IP 地址与 `ros_settings.txt` 中的一致。


# 开发

我们已在 `quad_controller_rl/` 中提供了启动代码，`src/quad_controller_rl/` 下也包含了所有的 Python 模块。此外，`notebooks/` 文件夹下是主要的项目 notebook。请查看这些文件，但此刻你并不需要更改其中的代码。首先请完成下面两个步骤（**建立**和**运行**），保证你的 ROS 正确安装。

## 建立

要想在 ROS 中运行你的代码，你首先需要建立它。这需要编译和连接项目所需的不同模块（“ROS 节点”）。幸运的是，该步骤只需要进行一次，因为对 Python 脚本进行的更改不需要重新编译。

- 打开你的 catkin workspace (`catkin_ws/`)：

```bash
$ cd ~/catkin_ws/
```

- 建立 ROS 节点：

```bash
$ catkin_make
```

- 启用命令行 tab 补全功能和其它实用的 ROS 应用：

```bash
$ source devel/setup.bash
```

## 运行

为了运行你的项目，请打开 `rl_controller.launch` 文件启动 ROS：

```bash
$ cd ~/catkin_ws/src/RL-Quadcopter/quad_controller_rl/launch
$ roslaunch quad_controller_rl rl_controller.launch
```

当不同节点启动时，你可以在终端中看见一些信息。此时你可以运行模拟器，这是一个单独的 Unity 应用（请注意，你必须先启动 ROS，再运行模拟器）。在模拟器初始化完成后，你将在 ROS 终端中看见其他信息，表示每隔几秒飞行器都会进入新的阶段。当飞行器接收到来自智能体的控制输入时，模拟器中飞行器的螺旋桨应当开始转动，并且这应当在每个阶段开始时重置。


小贴士：根据默认设置，每当你想要运行飞行器的模拟器时，你都需要按步骤建立和运行。如果你不想重复这两个步骤，可以编辑 `quad_controller_rl/scripts/drone_sim` 脚本，输入运行模拟器应用的命令。[这里](https://discussions.udacity.com/t/importerror-when-running-roslaunc-quad-controller-rl-rl-controller-launch/569530/2)有一个范例。接着，ROS 将自动启动模拟器。


_请注意：如果你想了解更多有关 ROS 的信息，以及如何将它应用于机器学习的应用中，你可以加入优达学城的 [机器人开发纳米学位](https://cn.udacity.com/course/robotics-nanodegree--nd209)并学习 [ROS 基础知识]( https://classroom.udacity.com/nanodegrees/nd209/parts/af07ae99-7d69-4b45-ab98-3fde8b576a16)模块。_

## 实现

一旦你确定 ROS 与模拟器都能顺利运行，并且相互连通，请尝试修改 `agents/policy_search.py` 中的代码（比如添加一条 `print` 语句）。这是一个默认运行的智能体模板。每当你做出改变，你都需要停止模拟器（在模拟器窗口下按 `Esc` 键），并关闭 ROS（在终端内按 `Ctrl+C` 键）。保存你的修改，并再次 `roslaunch`。

现在你应该准备好编写代码了！打开项目的 notebook 查看更多说明（假设你在你的 catkin workspace 中）：

```bash
$ jupyter notebook src/RL-Quadcopter/quad_controller_rl/notebooks/RL-Quadcopter.ipynb
```

# 提交

请按要求完成 notebook。完成之后，请将 notebook 保存/导出为 PDF 或 HTML 文件。这将作为你的项目报告。

如果你加入了优达学城的纳米学位，你可以在完成项目后提交审阅。你上传的压缩文件应包含以下内容：

- `RL-Quadcopter.ipynb`：项目的notebook，所有部分均已完成。
- `RL-Quadcopter.pdf` / `RL-Quadcopter.html`：PDF 或 HTML 报告（可以是 notebook 的导出文件）
- `quad_controller_rl/`：Python 包，以下分包中包含你的代码：
  - `tasks/`：本项目要求的每个任务的实现。
  - `agents/`：你为相应任务实现的智能体。
  - 项目中使用的其他辅助代码或文件。

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>. Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.
