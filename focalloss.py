import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class FocalLoss(nn.Module):  
    def __init__(self, alpha=1, gamma=2, reduction='mean'):  
        """  
        :param alpha: 平衡正负样本的权重因子  
        :param gamma: 减少易分类样本权重的调节指数  
        :param reduction: 指定应用于输出的归约方式. "none" | "mean" | "sum".  
                          "none": 不进行归约,  
                          "mean": 将得分求和后除以元素个数得到平均值,  
                          "sum": 将得分求和.  
        """  
        super(FocalLoss, self).__init__()  
        self.alpha = alpha  
        self.gamma = gamma  
        self.reduction = reduction  
  
    def forward(self, output, target):  
        """  
        :param output: 预测概率. 形状应为 (batch_size, h, w), 值应在 0 到 1 之间.  
        :param target: 真实标签. 形状应为 (batch_size, h, w), 值为类别索引.  
        :return: 计算得到的损失.  
        """  
        # 将输出转换为 logits（如果它们已经是概率，则可以通过 logit = -log(1 - p) for p in [0, 1] 的逆变换得到，但这里假设是概率）  
        # 但由于 Focal Loss 通常用于多分类的 softmax 输出后的概率，而这里看起来是二分类的概率图，所以我们直接使用概率  
        # 如果是多分类，则需要使用 one-hot 编码的 target 和 softmax 的 output，并计算每个类别的 Focal Loss，然后求和或平均  
        # 但这里我们假设是二分类，并且 output 是 sigmoid 后的输出  
  
        # 将 target 转换为与 output 相同的形状和类型的张量，但值为 0 或 1（对于二分类）  
        target = target.type_as(output).float()  
  
        # 计算交叉熵部分 -log(p_t)  
        BCE_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')  
  
        # 计算 p_t  
        pt = torch.exp(-BCE_loss)  # 因为 BCE_loss = -log(p_t)，所以 pt = exp(-BCE_loss) = p_t  
  
        # 计算 Focal Loss  
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  
  
        if self.reduction == 'mean':  
            return F_loss.mean()  
        elif self.reduction == 'sum':  
            return F_loss.sum()  
        else:  
            return F_loss  
  
# 示例用法  
batch_size = 4  
height = 32  
width = 32  
output = torch.randn(batch_size, height, width, requires_grad=True)  # 预测概率  
target = torch.randint(0, 2, (batch_size, height, width)).float()  # 真实标签（二分类）  
  
# 注意：这里 output 应该是 sigmoid 后的输出，但在这个例子中我们直接生成了它  
# 在实际应用中，你可能会有一个模型输出原始 logits，然后你需要对它们应用 sigmoid 函数  
# output = torch.sigmoid(model_output)  # 假设 model_output 是模型的原始输出（logits）  
  
# 创建 Focal Loss 实例并计算损失  
focal_loss = FocalLoss(alpha=1, gamma=2, reduction='mean')  
loss = focal_loss(output, target)  
print(loss)