import torch
from torch import nn, optim
from itertools import chain
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm

# import os
# import json
# import glob
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torch.utils import data
# import torchvision.transforms as T

# def plot_spectrogram(S, title=None, sr=16000, fmax=8000):
#     plt.figure(figsize=(10, 4))
#     S_dB = librosa.power_to_db(S, ref=np.max)
#     librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=fmax)
#     plt.colorbar(format='%+2.0f dB')
#     plt.tight_layout()
#     plt.title(title)
#     plt.show()



# def test_sample(encoder, decoders, discriminator, test_loader, device, target_domain):
#     loss_fn = nn.MSELoss()
    
#     for i, (source_audio, augmented_source_audio, source_domain) in enumerate(test_loader):

#         with torch.no_grad():
#             # Move data to correct device
#             print(type(source_audio))
#             source_audio = source_audio*10
#             source_audio = source_audio.to(device)
#             augmented_source_audio = augmented_source_audio.to(device)
#             source_domain = source_domain.to(device)

#             # Select decoder based on target domain
            
#             print("source domain: ",source_domain[0].item())
#             decoder = decoders[source_domain[0].item()]
            
#             if source_domain[0].item()==0:
#                 decoder2 = decoders[1]
#                 print("target domain: ",1)
#             else:
#                 decoder2 = decoders[0]
#                 print("target domain: ",0)
#             # Encode the augmented source audio
#             latent_vector = encoder(source_audio)

#             # Decode the latent vector
#             decoded_audio = decoder(latent_vector)
#             translated_audio = decoder2(latent_vector)

#             # Calculate the cross entropy loss
#             loss = F.sigmoid(loss_fn(decoded_audio, source_audio))

# #             print("loss source-decoded: ", loss)

#             plot_spectrogram(source_audio[0].squeeze(dim=0), title="source")
#             plot_spectrogram(decoded_audio[0].squeeze(dim=0), title="decoded")
#             plot_spectrogram(translated_audio[0].squeeze(dim=0), title="translated")
            
#             break
#             if i>5:
#                 break
            
#     return source_audio, decoded_audio, translated_audio

# def train_discriminator(latent_vector, discriminator, source_domain):
#     # Predict the source domain of the latent vector
#     pred_domain = discriminator(latent_vector)

#     # Calculate loss
#     loss = discriminator.loss_fn(pred_domain, source_domain)

#     # Set gradient to zero and perform backpropagation
#     discriminator.optimizer.zero_grad()
#     loss.backward()

#     # Perform weight optimization
#     discriminator.optimizer.step()

#     return loss.item()


def train_epoch(encoder, decoders, discriminator, train_loader, device, plotter, lr=0.01):
    loss_fn = nn.MSELoss()
    epoch_loss = 0

    for i, (source_audio, augmented_source_audio, source_domain) in enumerate(train_loader): #in enumerate(tqdm(train_loader))
#         assert((source_domain[0] == source_domain).all())
        # Move data to correct device
        source_audio = source_audio.to(device)
        augmented_source_audio = augmented_source_audio.to(device)
        source_domain = source_domain.to(device)

        # Select decoder based on domain
        batch_domain = source_domain[0].item()
#         print("source domain: ",source_domain[0].item())
#         print(type(decoders))
        if type(decoders)!=dict:
            decoder = decoders
        else:
            decoder = decoders[batch_domain]

        # Create optimizer
        optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=lr)

        # Train the discriminator
        latent_vector = encoder(source_audio)
#         train_d<iscriminator(latent_vector, discriminator, source_domain)

        # Train the Encoder-Decoder pair
        # Encode the augmented source audio
#         latent_vector = encoder(augmented_source_audio)

        # Decode the latent vector
        decoded_audio = decoder(latent_vector)

        # Calculate discriminator loss
        pred_domain = discriminator(latent_vector)
#         discriminator_loss = discriminator.loss_fn(pred_domain, source_domain)

        # Calculate the cross entropy loss
        loss = loss_fn(decoded_audio, source_audio)

        # Subtract loss from discriminator
        loss_ed = loss.item()
        loss = loss# - (discriminator.weight * discriminator_loss)

        # Perform backpropagation
        optimizer.zero_grad()
#         loss.backward()

        # Perform optimization
        optimizer.step()

        plotter.add_losses(loss.item(), loss_ed, 0)#discriminator_loss.item())
        plotter.plot()

#     return epoch_loss


def eval_epoch(encoder, decoder, discriminator, val_loader, device):
    loss_fn = nn.MSELoss()
    losses = []
    n_correct = 0

    for source_audio, augmented_source_audio, source_domain in val_loader:
        source_audio = source_audio.to(device)
        augmented_source_audio = augmented_source_audio.to(device)
        source_domain = source_domain.to(device)

        with torch.no_grad():
            latent_vector = encoder(source_audio)
            decoded_audio = decoder(latent_vector)
            # Calculate discriminator loss
            pred_domain = discriminator(latent_vector)
            discriminator_loss = discriminator.loss_fn(pred_domain, source_domain)

            hard_preds = pred_domain.argmax(dim=1)

            # Calculate the cross entropy loss
            loss = loss_fn(decoded_audio, source_audio)

            # Subtract loss from discriminator
            total_loss = loss - (discriminator.weight * discriminator_loss)

            losses.append(total_loss)
            n_correct += torch.sum(hard_preds == source_domain).item()

        val_accuracy = n_correct/len(val_loader.dataset)
        val_avg_loss = sum(losses)/len(losses)

    return val_accuracy, val_avg_loss


def train(encoder, decoders, discriminator, num_epochs, domains, train_loader, val_loader, device, plotter, lr=0.01):
    # Discriminator training parameters
    discriminator.loss_fn = nn.CrossEntropyLoss()
    discriminator.optimizer = optim.Adam(discriminator.parameters())
    discriminator.weight = 0.1

    for epoch in range(num_epochs):
        train_loss = train_epoch(encoder, decoders, discriminator, train_loader, device, plotter,lr)
#         eval_epoch(encoder, decoders, discriminator, val_loader, device)
#         print("Epoch {}: train loss: {}, val loss: {}, val accuracy: {}".format(epoch, train_loss, 0, 0))
