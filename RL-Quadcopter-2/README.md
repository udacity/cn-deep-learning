# DeepRL 四轴飞行器控制器

_指导四轴飞行器学会飞行！_

在本次项目中，你将设计一个深度强化学习智能体，来控制几个四轴飞行器的飞行任务，包括起飞、悬停和着陆。

## 项目说明

1. 复制代码库，并浏览下载文件夹。

```
git clone https://github.com/udacity/RL-Quadcopter-2.git
cd RL-Quadcopter-2
```

2. 创建并激活一个新的环境。

```
conda create -n quadcop python=3.6 matplotlib numpy pandas
source activate quadcop
```

3. 为 `quadcop` 环境创建一个 [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html)。 
```
python -m ipykernel install --user --name quadcop --display-name "quadcop"
```

4. 打开 notebook。
```
jupyter notebook Quadcopter_Project.ipynb
```

5. 在运行代码之前，请使用 drop-down 菜单（**Kernel > Change kernel > quadcop**） 修改 kernel 以适应 `quadcop` 环境。接着请按照 notebook 中的说明进行操作。

6. 为了完成本项目，你也许还需要安装额外的 pip 包。请查看代码库中的 `requirements.txt` 文件，以了解运行项目所需的包。
