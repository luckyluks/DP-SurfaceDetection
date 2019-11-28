import torch
from torch import nn, optim
from itertools import chain
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm



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
