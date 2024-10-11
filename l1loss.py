import torch 
  
def l1_loss(output, target):  
    
      
    # 计算差的绝对值  
    absolute_difference = torch.abs(output - target)  
      
    # 计算整个batch的平均L1损失  
    loss = torch.mean(absolute_difference) *100*100 
      
    return loss  
  
