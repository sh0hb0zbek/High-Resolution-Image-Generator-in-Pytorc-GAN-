import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torch.optim as optim
import torchvision.utils as vutils
import time

from save_models import save_model
import numpy as np



def training(
        generator,
        discriminator,
        real_image_loader,
        num_epoch,
        batch_size,
        writer: SummaryWriter,
        learning_rate: float = 0.002,
        device: str = "cuda",
        beta_1: float = 0.0,
        beta_2: float = 0.99,
        dis_reg_interval: int = 16):
    noise_size = generator.input_shape[-1]
    constant_noise = torch.normal(0.0, 1.0, (16, noise_size)).to(device)
    
    gen_optimizer = optim.Adam(params=generator.parameters(),
                               lr=learning_rate,
                               betas=(beta_1, beta_2))
    lazy_ratio = dis_reg_interval / (dis_reg_interval + 1)
    dis_optimizer = optim.Adam(params=discriminator.parameters(),
                               lr=learning_rate*lazy_ratio,
                               betas=(beta_1**lazy_ratio, beta_2**lazy_ratio))
    num_batch = len(real_image_loader)
    # true_labels = torch.ones(batch_size, dtype=torch.float32, device=device)
    # fake_labels = torch.zeros(batch_size, dtype=torch.float32, device=device)
    gen_losses = []
    dis_losses = []
    real_losses = []
    fake_losses = []
    dis_reg_losses = []

    # Define the reduce across batch function
    def reduce_across_batch(x):
        return x.sum() / batch_size


    def take_g_step(fake_images) -> dict:
        fake_classifications = discriminator(fake_images)
        loss = reduce_across_batch(F.softplus(-fake_classifications))

        gen_optimizer.zero_grad()
        loss.backward()
        gen_optimizer.step()

        return loss.item()

    def take_d_classification_step(real_images) -> dict:
        noise = torch.normal(0.0, 1.0, (batch_size, noise_size)).to(device)
        fake_images = generator(noise)
        real_classifications = discriminator(real_images)
        fake_classifications = discriminator(fake_images)

        real_loss = reduce_across_batch(F.softplus(-real_classifications))
        fake_loss = reduce_across_batch(F.softplus(fake_classifications))
        d_loss = real_loss + fake_loss

        dis_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        dis_optimizer.step()

        return d_loss.item(), real_loss.item(), fake_loss.item()

    def take_d_reg_step(real_images) -> dict:
        real_images.requires_grad = True
        real_classifications = discriminator(real_images)
        real_grads = torch.autograd.grad(outputs=torch.sum(real_classifications), inputs=real_images, create_graph=True)[0]
        gradient_loss = reduce_across_batch(torch.sum(real_grads.pow(2), dim=[1, 2, 3]))
        gradient_penalty_strength = 10. * 0.5 * dis_reg_interval
        gradient_penalty = gradient_loss * gradient_penalty_strength

        dis_optimizer.zero_grad()
        gradient_penalty.backward()
        dis_optimizer.step()

        return gradient_penalty.item()
    
    for epoch in range(num_epoch):
        start_time = time.time()
        for batch_i, real_images in enumerate(real_image_loader):
            noise = torch.normal(0.0, 1.0, (batch_size, noise_size)).to(device)
            fake_images = generator(noise)
            batch = real_images.to(device)
            # take a generator step
            g_loss = take_g_step(fake_images)

            # Take a discriminator classification step
            d_loss, real_loss, fake_loss = take_d_classification_step(batch)

            # Take a discriminator regularization step
            if batch_i % dis_reg_interval == 0:
                dis_reg_loss = take_d_reg_step(batch)
                dis_reg_losses.append(dis_reg_loss)
            
            gen_losses.append(g_loss)
            dis_losses.append(d_loss)
            real_losses.append(real_loss)
            fake_losses.append(fake_loss)
            
            print(f"Epoch: {epoch+1:02d}/{num_epoch}    iter: {batch_i}/{num_batch}    g_loss: {np.mean(gen_losses):.5f}    d_loss: {np.mean(dis_losses):.5f}    r_loss: {np.mean(real_losses):.5f}    f_loss: {np.mean(fake_losses):.5f}    reg_loss: {np.mean(dis_reg_losses):.5f}    T: {time.time() - start_time:.3f} s")#, end="\r")
            # print(f"Epoch: {epoch+1:02d}/{num_epoch}")
            # print(f"iter: {batch_i}/{num_batch}")
            # print(f"g_loss: {np.mean(gen_losses):.5f}")
            # print(f"d_loss: {np.mean(dis_losses):.5f}")
            # print(f"r_loss: {np.mean(real_losses):.5f}")
            # print(f"f_loss: {np.mean(fake_losses):.5f}")
            # print(f"reg_loss: {np.mean(dis_reg_losses):.5f}")
            # print(f"T: {time.time() - start_time:.3f} s")#, end="\r")
            writer.add_scalar("gen_loss", np.mean(gen_losses), batch_i + 1 + num_batch * epoch)
            writer.add_scalar("dis_loss", np.mean(dis_losses), batch_i + 1 + num_batch * epoch)
            writer.add_scalar("real_loss", np.mean(real_losses), batch_i + 1 + num_batch * epoch)
            writer.add_scalar("fake_loss", np.mean(fake_losses), batch_i + 1 + num_batch * epoch)
            writer.add_scalar("dis_reg_loss", np.mean(dis_reg_losses), batch_i + 1 + num_batch * epoch)
        writer.add_image("fake", vutils.make_grid(generator(constant_noise).data, normalize=True), epoch + 1)
        writer.add_image("real", vutils.make_grid(batch.data[:16], normalize=True), epoch + 1)
        save_model(generator, "/shared/home/iseiot/12184741/GAN/GAN_2024/models/generator/generator", epoch+1)
        save_model(discriminator, "/shared/home/iseiot/12184741/GAN/GAN_2024/models/discriminator/discriminator", epoch+1)
        
        
            