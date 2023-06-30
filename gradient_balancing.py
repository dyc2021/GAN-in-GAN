import torch
import torch.nn as nn


class GradientBalancing(nn.Module):
    def __init__(self, losses_num):
        super().__init__()
        self.losses_num = losses_num
        self.e = nn.Parameter(torch.exp(torch.ones([1])), requires_grad=False)
        self.s_t = nn.Parameter(torch.ones(losses_num).squeeze(), requires_grad=True)

    def balance_losses(self, losses: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(losses) != self.losses_num:
            raise Exception("the gradient balancing module receives invalid input.")
        final_losses = []
        for i, loss in enumerate(losses):
            final_losses.append(loss * self.e.pow(self.s_t[i]) - self.s_t[i])

        return final_losses


if __name__ == "__main__":
    gradient_balance_module = GradientBalancing(losses_num=4)
    for i in gradient_balance_module.parameters():
        print(i)
