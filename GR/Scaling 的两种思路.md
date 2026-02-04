
ScalingLaw的本质就是模型的性能随着模型大小/训练数据量/计算量等的增加而增加。

## OneRec的scaling思路（生成式的思路）

从模型的整个链路出发，从训练侧和推理侧的角度出发。


## RankMixer的scaling思路（判别式的思路）

<mark>从特征交叉和self-attention的角度出发。</mark>

传统的特征交叉通过叠加FM或者LCB来完成embedding的特征融合，而rankmixer则考虑在tokenizer部分完成特征交叉。