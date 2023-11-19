
- ==“train_resnet.py”== 有两个强制性参数， 
    1。“data_folder”（工作区的路径）。
    2 “output_folder”（要保存训练模型的文件夹）。
    您可能想要设置的另一个可选参数是“--model_path”，在 ImageNet 或 PLACE365 上预先训练用于分类，如果未指定，它将从头开始训练。

&
- 在对SALCON数据集上的主干（来自ImageNet的resnet50）进行微调后，**我们可以通过训练解码器来组合多个显著性模型（ImageNet和PLACE）**。在这种情况下，我们需要另外两个强制性参数，即 imagenet 和 place 的模型路径。（您可以稍微更改代码以合并更多代码以获得更广阔的视野。

- timm这个库里面可以直接把一个图片划分一个patch快
- 隐含表征   按之前的顺序拼起来 unshuffle
- masking 16*16*3 随机采样 
    - 先生成 token embed = input embed + position embed 变成一个序列
    - shuffle 一下---就可以按个来取了，把他masked掉 ，可以还原。 可以直接通过代码来。  
- encoder 
    - 做分类任务就编码器就可以做
    - 有87.6的准确率
- decoder 完整的长度 编码过后的encoder masked 要返回 怎么返回？
    - PE 还是会有 
    - 预测一个个masked 的像素值 
- 最后一层还有Liner
    - 把这个映射到相应的masked位置上

