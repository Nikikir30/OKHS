import torch
import torch.nn as nn
from scipy.special import gamma, roots_jacobi

class DeepFractionalDMDLoss(nn.Module):
    """
    Вычисляет интегральную невязку в латентном пространстве \tilde{H}.
    Ожидает на вход уже интерполированные и закодированные тензоры.
    """
    def __init__(self, latent_dim: int, q: float, n_quad_points: int = 20, num_tk_samples: int = 5, device: str = 'cpu'):
        super().__init__()
        self.q = q
        self.latent_dim = latent_dim
        self.num_tk_samples = num_tk_samples
        
        self.W = nn.Parameter(-1 * torch.eye(latent_dim, dtype=torch.float64, device=device) + 
                              torch.randn(latent_dim, latent_dim, dtype=torch.float64, device=device) * 0.1)

        nodes, weights = roots_jacobi(n_quad_points, q - 1, 0)
        self.register_buffer('nodes', torch.tensor(nodes, dtype=torch.float64, device=device))
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float64, device=device))
        self.c_q = 1.0 / gamma(q)

    def get_collocation_nodes(self, T_norm: torch.Tensor) -> tuple:
        """
        Генерирует целевые узлы коллокации в нормированном времени tau.
        Используется во внешнем цикле обучения для передачи в TimeGridManager.
        
        Параметры:
            T_norm (torch.Tensor): Нормированные длительности батча (N,)
        Возвращает:
            t_k (torch.Tensor): Концы подынтервалов (N, num_tk_samples)
            tau_nodes (torch.Tensor): Узлы квадратуры (N, num_tk_samples, Q)
        """
        device = T_norm.device
        
        steps = torch.linspace(1.0 / self.num_tk_samples, 1.0, self.num_tk_samples, device=device)
        t_k = steps.unsqueeze(0) * T_norm.view(-1, 1)

        # Аффинное отображение узлов Гаусса-Якоби [-1, 1] -> [0, t_k]
        tau_nodes = (t_k.unsqueeze(-1) / 2.0) * (self.nodes.view(1, 1, -1) + 1.0)
        
        return t_k, tau_nodes

    def forward(self, z_start: torch.Tensor, z_tk: torch.Tensor, z_nodes: torch.Tensor, t_k: torch.Tensor) -> torch.Tensor:
        """
        Метод принимает готовые латентные представления и вычисляет 
        штраф за невязку с оператором Лиувилля.
        
        Параметры:
            z_start (torch.Tensor): Точки в tau=0 (N, m)
            z_tk (torch.Tensor): Точки на концах интервалов t_k (N, S, m)
            z_nodes (torch.Tensor): Точки в узлах квадратуры (N, S, Q, m)
            t_k (torch.Tensor): Длины интервалов для масштабирования интеграла (N, S)
        """
        scale = self.c_q * ((t_k / 2.0) ** self.q)  # (N, S)

        integral_sum = torch.einsum('q, nsqm -> nsm', self.weights, z_nodes)
        fractional_integral = scale.unsqueeze(-1) * integral_sum
        
        w_action = torch.matmul(fractional_integral, self.W.T)
        
        boundary_diff = z_tk - z_start.unsqueeze(1) # (N, S, m) - (N, 1, m)
        
        residual = boundary_diff - w_action
        loss_adjoint = torch.mean(torch.linalg.norm(residual, dim=-1)**2)
        
        return loss_adjoint