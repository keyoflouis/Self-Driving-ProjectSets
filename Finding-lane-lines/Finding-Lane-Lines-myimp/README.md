

### 配置 PyCharm 使用 Conda 环境

#### 步骤 1：打开 PyCharm 中的 Python 项目

打开 PyCharm。

选择或创建一个 Python 项目。

-> `File`。

-> `Settings`

-> `Project: [项目名称]`。

-> `Python Interpreter`。

#### 步骤 2：添加 Conda 解释器

-> `Add Interpreter`。

选择以下两种方式之一：

##### 方式一：生成新的 Conda 环境

- 选择 `Generate new`。

- 使用 Type 选中的解释器创建一个环境，这相当于执行命令：
  
  `conda create --name test python=3.9`。

- 填写 `condabin\conda.bat` 的绝对路径
  
  （如果更改了安装路径，且没有添加环境变量，系统无法自动找到Anaconda的condabin\conda.bat路径,，也无法找到你创建的环境）。

##### 方式二：选择已存在的 Conda 环境

- 选择 `Select existing`。

- 填写  `condabin\conda.bat` 的绝对路径。

- 在 `Environment` 中选择已经创建好的环境。
  
  （如果更改了安装路径，且没有添加环境变量，系统无法自动找到Anaconda的condabin\conda.bat路径,，也无法找到你创建的环境）。

---