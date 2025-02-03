### 代码流程

首先，正向传播

然后使用反转列表，反向传播。

---

##### MSE的传播

首先对实际值，预测值求得梯度，存入字典类型中 

`self.gradients[self.inbound_nodes[0]]`

`self.gradients[self.inbound_nodes[1]]`

##### Sigmoid的传播

- （根据前向传播输入的节点）创建一个空的 `self.gradients` 字典,用于存储**损失函数**相对于输入值的梯度

- （链式法则通过反向传播来实现）求得sigmoid的梯度（预测值对于y（线性函数输出值））的梯度），读取MSE对预测值的梯度。

- 根据链式法则，可以得到MSE相对于输入sigmoid的y（线性函数输出值）的梯度，`MSE对预测值的梯度`乘以`预测值对于x的梯度`，` grad_cost * sigmoid_derivative`

##### Liner的传播

- 根据Input，创建一个空的 `self.gradients` 字典

- 链式法则的过程，但不一样的是，这里涉及到矩阵转置（为了使得矩阵链式法则运算正确）

##### Input的传播

- 没有输入节点，因此创建一个字典，记录 `self:0` 

- 只需加上，上一层的梯度

<br>

### 关于矩阵转置的解释

在神经网络的反向传播过程中，**Linear层（全连接层**的梯度计算涉及到矩阵运算，尤其是矩阵的转置操作。为了更清楚地理解这一点，我们可以通过一个具体的例子来解释。

#### 例子

假设我们有一个简单的Linear层，其输入 $ x $ 是一个2维向量，权重 $ W $ 是一个2x3的矩阵，偏置 $ b $ 是一个3维向量。输出 $ y $ 是一个3维向量。

#### 正向传播

1. **输入**：
   
   $$
   x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
   $$

2. **权重**：
   
   $$
   W = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \\ w_{31} & w_{32} \end{bmatrix}
   $$

3. **偏置**：
   
   $$
   b = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}
   $$

4. **输出**：
   
   $$
   y = Wx + b = \begin{bmatrix} w_{11}x_1 + w_{12}x_2 + b_1 \\ w_{21}x_1 + w_{22}x_2 + b_2 \\ w_{31}x_1 + w_{32}x_2 + b_3 \end{bmatrix}
   $$

#### 反向传播

在反向传播过程中，我们需要计算损失函数对权重 $ W $ 和输入 $ x $ 的梯度。

1. **损失函数对输出 $ y $ 的梯度**：
   假设损失函数对 $ y $ 的梯度为：
   
   $$
   \frac{\partial \text{MSE}}{\partial y} = \begin{bmatrix} \frac{\partial \text{MSE}}{\partial y_1} \\ \frac{\partial \text{MSE}}{\partial y_2} \\ \frac{\partial \text{MSE}}{\partial y_3} \end{bmatrix}
   $$

2. **损失函数对权重 $ W $ 的梯度**：
   根据链式法则，损失函数对权重 $ W $ 的梯度为：
   
   $$
   \frac{\partial \text{MSE}}{\partial W} = \frac{\partial \text{MSE}}{\partial y} \cdot x^T
   $$
   
   这里 $ x^T $ 是输入 $ x $ 的转置。具体计算如下：
   
   $$
   x^T = \begin{bmatrix} x_1 & x_2 \end{bmatrix}
   $$
   
   $$
   \frac{\partial \text{MSE}}{\partial W} = \begin{bmatrix} \frac{\partial \text{MSE}}{\partial y_1} \\ \frac{\partial \text{MSE}}{\partial y_2} \\ \frac{\partial \text{MSE}}{\partial y_3} \end{bmatrix} \cdot \begin{bmatrix} x_1 & x_2 \end{bmatrix} = \begin{bmatrix} \frac{\partial \text{MSE}}{\partial y_1} x_1 & \frac{\partial \text{MSE}}{\partial y_1} x_2 \\ \frac{\partial \text{MSE}}{\partial y_2} x_1 & \frac{\partial \text{MSE}}{\partial y_2} x_2 \\ \frac{\partial \text{MSE}}{\partial y_3} x_1 & \frac{\partial \text{MSE}}{\partial y_3} x_2 \end{bmatrix}
   $$

3. **损失函数对输入 $ x $ 的梯度**：
   根据链式法则，损失函数对输入 $ x $ 的梯度为：
   
   $$
   \frac{\partial \text{MSE}}{\partial x} = W^T \cdot \frac{\partial \text{MSE}}{\partial y}
   $$
   
   这里 $ W^T $ 是权重 $ W $ 的转置。具体计算如下：
   
   $$
   W^T = \begin{bmatrix} w_{11} & w_{21} & w_{31} \\ w_{12} & w_{22} & w_{32} \end{bmatrix}
   $$
   
   $$
   \frac{\partial \text{MSE}}{\partial x} = \begin{bmatrix} w_{11} & w_{21} & w_{31} \\ w_{12} & w_{22} & w_{32} \end{bmatrix} \cdot \begin{bmatrix} \frac{\partial \text{MSE}}{\partial y_1} \\ \frac{\partial \text{MSE}}{\partial y_2} \\ \frac{\partial \text{MSE}}{\partial y_3} \end{bmatrix} = \begin{bmatrix} w_{11} \frac{\partial \text{MSE}}{\partial y_1} + w_{21} \frac{\partial \text{MSE}}{\partial y_2} + w_{31} \frac{\partial \text{MSE}}{\partial y_3} \\ w_{12} \frac{\partial \text{MSE}}{\partial y_1} + w_{22} \frac{\partial \text{MSE}}{\partial y_2} + w_{32} \frac{\partial \text{MSE}}{\partial y_3} \end{bmatrix}
   $$

### 总结

在这个例子中，我们通过具体的矩阵运算展示了反向传播过程中如何计算损失函数对权重 $ W $ 和输入 $ x $ 的梯度。关键点在于理解矩阵的转置操作（如 $ x^T $ 和 $ W^T $）在梯度计算中的作用。希望这个例子能帮助你更好地理解反向传播的机制。