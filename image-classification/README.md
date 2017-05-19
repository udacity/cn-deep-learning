## 在 floydhub.com 上运行优达学城深度学习基石纳米学位图像分类项目


1. 在 [floydhub.com](https://www.floydhub.com) 上创建一个帐户（别忘了确认电子邮件）。你将自动获得 100 个免费 GPU 小时。

2. 在你的计算机上运行 `floyd` 命令：

        pip install -U floyd-cli

    即使你之前已经安装了 `floyd-cli`，也要执行这一步。确保你安装的是最新版本（它的开发速度很快！）。


3. 用此命令绑定你的 Floyd 账号：

        floyd login

    （系统会在浏览器中打开一个具有身份验证令牌的页面，你需要将该令牌复制到你的终端里）

4. 克隆代码库:

        git clone https://github.com/ludwiktrammer/deep-learning.git

    注意：这个代码库与优达学城的代码库之间有些许差别。你可以在 [README](https://github.com/ludwiktrammer/deep-learning/tree/master/image-classification#how-is-this-repository-different-from-the-original) 找到详细说明。要使用这份说明一步步操作，我们建议你使用 ludwiktrammer 的代码库。

5. 进入图像分类项目文件夹：

        cd image-classification

6. 初始化 Floyd 项目：

        floyd init dlnd_image_classification

7. 运行项目：

        floyd run --gpu --env tensorflow --mode jupyter --data diSgciLH4WA7HpcHNasP9j

    这段命令的意思是：它将在有 GPU（`--gpu`）的机器上运行，使用 Tenserflow 环境（`--env tensorflow`），使用 Jupyter 记事本（`--mode jupyter`），且可用 Floyd 的内置的 cifar-10 数据集（`--data diSgciLH4WA7HpcHNasP9j`）。

8. 等待 Jupyter 记事本准备好，然后复制终端里显示的 URL（见 “path to jupyter notebook”）在浏览器中打开，你将看到该记事本。


9. 当你没有使用该记事本时，请记得关闭实验（experiment）。只要实验在运行（即使是在后台运行），就会消耗 GPU 时间，而你只有 100 小时的免费时间。你可以在 floyd.com 的“[Experiments](https://www.floydhub.com/experiments)”部分，或使用 `floyd stop` 命令停止实验：

        floyd stop ID

   （其中 ID 是当你运行该项目时，在终端里显示的 “RUN ID”。如果你找不到该 ID，可以在 floyd.com 的“[Experiments](https://www.floydhub.com/experiments)”部分找到。）

**重要提醒**：当你运行项目时，它将始终从头开始（即从计算机上的本地状态开始）。如果你在此前的运行中，对服务器上的  Jupiter 记事本上做了修改，这些更改将不会在之后的运行中生效。要永久保留这些更改，你需要将这些更改添加到本地项目文件夹中。运行记事本时，你可以直接从  Jupyter 菜单栏的 - *File / Download / Notebook* 下载记事本。下载完毕后，将本地的 `dlnd_image_classification.ipynb` 文件替换为新下载的文件即可。

如果你已经停止实验，依然可以使用 `floyd output` 命令下载文件：

    floyd output ID

   （其中 ID 是当你运行该项目时，在终端里显示的 “RUN ID”。如果你找不到该 ID，可以在 floyd.com 的“[Experiments](https://www.floydhub.com/experiments)”部分找到。）

只需运行上述命令，下载 `dlnd_image_classification.ipynb`，并将本地版本替换为新下载的文件即可。


## ludwiktrammer 代码库与[优达学城的原代码库](https://github.com/udacity/deep-learning)有何区别？

1. 添加了对 Floyds built-in cifar-10 数据集的支持。如果检测到该数据集，将直接使用，无需再下载任何文件。（[见此 commit](https://github.com/ludwiktrammer/deep-learning/commit/2e84ff7852905f154f1692f67ca15da28ac43149)，[详细了解 Floyd 提供的数据集](http://docs.floydhub.com/guides/datasets/)）

2. 添加了 `floyd_requirements.txt` 文档，自动处理另一个依赖问题。（[见此 commit](https://github.com/ludwiktrammer/deep-learning/commit/80b459411d4395dacf8f46be0b028c81858bd97a)，[详细了解 `.floyd_requirements.txt` 文档](http://docs.floydhub.com/home/installing_dependencies/)）

3. 添加了 `.floydignore` 文档，防止上传本地数据到 Floyd，以免浪费时间，甚至出现超时的问题。（([见此 commit](https://github.com/ludwiktrammer/deep-learning/commit/30d4b536b67366feef38425ce1406e969452717e)，[详细了解 `.floydignore` 文档](http://docs.floydhub.com/home/floyd_ignore/))

4. 添加了这个 README

