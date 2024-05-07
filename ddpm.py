import torch 


class DDPM:
    def __init__(self, beta_start=1e-4, beta_end=2e-2, steps=1000, image_size=16, color_channels=3, device='cpu'):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.steps = steps
        self.image_size = image_size
        self.device = device
        self.color_channels = color_channels
        
        self.beta = torch.linspace(beta_start, beta_end, steps).to(device)
        
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def forward_diffusion(self, x, t):
        sqrt_a_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_a_hat = torch.sqrt(1-self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_a_hat*x + sqrt_one_minus_a_hat*noise, noise
    
    def time_step_sampler(self, i):
        return torch.randint(low=1, high=self.steps, size=(i,))
    
    def backward_diffusion(self, model, num_images):
        model.eval()     
        with torch.no_grad():
            x = torch.randn((num_images, 3, self.image_size, self.image_size)).to(self.device)
            for i in reversed(range(1, self.steps)):
                t = (torch.ones(num_images)*i).long().to(self.device)
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 1:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)
                    
                sigma = torch.sqrt(beta)
                coef = (1 -  alpha) / (torch.sqrt(1 - alpha_hat))
                x = 1 / torch.sqrt(alpha) * (x - coef * model(x, t)) + sigma*z  
                
        model.train()
        x = (x.clamp(-1, 1))/2 + 0.5 
        return x 

        