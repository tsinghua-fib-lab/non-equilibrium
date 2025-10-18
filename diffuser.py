import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import DDPMScheduler

from model import NoiseModel


if __name__ == '__main__':

    # 2. Hyperparameters
    diffusion_N = 1000 # Total steps for scheduler's noise schedule
    training_epoch = 100
    lr = 1e-3
    device = "cuda:2"

    # 3. Instantiate Model and Scheduler
    model = NoiseModel().to(device)
    scheduler = DDPMScheduler(num_train_timesteps=diffusion_N, prediction_type='epsilon', clip_sample=False) # Scheduler defines noise schedule

    # 4. Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() # Standard for noise prediction

    # 5. Load data
    num_particles = 1000
    num_s = 5
    d = np.load(f'data/N{num_particles}_s{num_s}.npz')
    tr = d['trajectory']
    Ts = tr.shape[0]
    t_sim = np.linspace(0, 4.0*num_s, Ts, dtype=np.float32)

    p_flat = tr.reshape(-1, 2)
    t_flat = np.repeat(t_sim, num_particles).astype(np.float32)

    p_t = torch.tensor(p_flat, dtype=torch.float).to(device)
    t_t = torch.tensor(t_flat, dtype=torch.float).to(device)

    from torch.utils.data import Dataset, DataLoader

    class PD(Dataset):
        def __init__(self, p, t): self.p, self.t = p, t
        def __len__(self): return len(self.p)
        def __getitem__(self, i): return self.p[i], self.t[i]

    dataset = PD(p_t, t_t)
    bs = 512
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

    # 6. Train
    print(f"Train on {device} ({len(dataset)} samples, bs {bs})...")

    for epoch in range(training_epoch):
        model.train()
        loss_sum = 0.0
        for cx, tdyn in dataloader:
            sdiff = torch.randint(0, diffusion_N, (bs,), device=device).float()
            noise = torch.randn_like(cx)
            noisy_x = scheduler.add_noise(cx, noise, sdiff.long())

            pred = model(noisy_x, tdyn, sdiff)

            loss = loss_fn(pred, noise)
            # loss = loss_fn(pred, cx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * cx.size(0)

        avg_l = loss_sum / len(dataset)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{training_epoch}, Loss: {avg_l:.5f}")

    os.makedirs(f'log/N{num_particles}_s{num_s}/diffusion/', exist_ok=True)
    torch.save(model.state_dict(), f'log/N{num_particles}_s{num_s}/diffusion/diffusion.pt')