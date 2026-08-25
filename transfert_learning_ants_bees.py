# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from PIL import Image
from tempfile import TemporaryDirectory
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


# ============================================================
# Configuration
# ============================================================

cudnn.benchmark = True
plt.ion()

url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
project_dir = Path(__file__).resolve().parent
zip_file = project_dir / "hymenoptera_data.zip"
data_dir = project_dir / "hymenoptera_data"


# Le device est compatible avec CUDA, MPS (macOS) et CPU.
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)
cudnn.benchmark = device.type == "cuda"


# ============================================================
# Téléchargement et décompression du dataset
# ============================================================

if not data_dir.exists():
    print("Téléchargement du dataset...")

    urlretrieve(url, zip_file)

    print("\nDécompression du dataset...")

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(project_dir)

    print("Dataset téléchargé et décompressé !")
else:
    print("Dataset déjà présent, téléchargement ignoré.")


# ============================================================
# Data augmentation et normalisation
# ============================================================

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
}


# ============================================================
# Chargement des datasets
# ============================================================

image_datasets = {
    x: datasets.ImageFolder(
        data_dir / x,
        data_transforms[x]
    )
    for x in ['train', 'val']
}


# ============================================================
# DataLoaders
# IMPORTANT : num_workers=0 pour Mac / MPS
# ============================================================

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=4,
        shuffle=x == 'train',
        num_workers=0
    )
    for x in ['train', 'val']
}


# ============================================================
# Informations sur le dataset
# ============================================================

dataset_sizes = {
    x: len(image_datasets[x])
    for x in ['train', 'val']
}

class_names = image_datasets['train'].classes


print("Classes :", class_names)
print("Taille train :", dataset_sizes['train'])
print("Taille validation :", dataset_sizes['val'])


print(f"Using {device} device")


# ============================================================
# Visualisation de quelques images
# ============================================================

def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)


# ============================================================
# Récupération d'un batch
# ============================================================

inputs, classes = next(iter(dataloaders['train']))

out = torchvision.utils.make_grid(inputs)

imshow(
    out,
    title=[class_names[x] for x in classes]
)


# ============================================================
# Fonction d'entraînement
# ============================================================

def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25
):

    since = time.time()

    # Répertoire temporaire pour sauvegarder le meilleur modèle
    with TemporaryDirectory() as tempdir:

        best_model_params_path = Path(tempdir) / 'best_model_params.pt'

        torch.save(
            model.state_dict(),
            best_model_params_path
        )

        best_acc = 0.0

        # ----------------------------------------------------
        # Boucle sur les epochs
        # ----------------------------------------------------

        for epoch in range(num_epochs):

            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # ------------------------------------------------
            # Phase train / validation
            # ------------------------------------------------

            for phase in ['train', 'val']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0

                # IMPORTANT :
                # On garde ce compteur en Python et non
                # comme Tensor MPS.
                running_corrects = 0

                # ------------------------------------------------
                # Parcours des données
                # ------------------------------------------------

                for inputs, labels in dataloaders[phase]:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Remise à zéro des gradients
                    optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(
                        phase == 'train'
                    ):

                        outputs = model(inputs)

                        _, preds = torch.max(
                            outputs,
                            1
                        )

                        loss = criterion(
                            outputs,
                            labels
                        )

                        # Backward uniquement en train
                        if phase == 'train':

                            loss.backward()

                            optimizer.step()

                    # ------------------------------------------------
                    # Statistiques
                    # ------------------------------------------------

                    running_loss += (
                        loss.item() * inputs.size(0)
                    )

                    # CORRECTION MPS :
                    # .item() convertit le Tensor MPS
                    # en nombre Python.
                    running_corrects += torch.sum(
                        preds == labels.data
                    ).item()

                # Mise à jour du learning rate
                if phase == 'train':
                    scheduler.step()

                # ------------------------------------------------
                # Calcul des métriques
                # ------------------------------------------------

                epoch_loss = (
                    running_loss /
                    dataset_sizes[phase]
                )

                # CORRECTION MPS :
                # plus de .double()
                epoch_acc = (
                    running_corrects /
                    dataset_sizes[phase]
                )

                print(
                    f'{phase} Loss: {epoch_loss:.4f} '
                    f'Acc: {epoch_acc:.4f}'
                )

                # ------------------------------------------------
                # Sauvegarde du meilleur modèle
                # ------------------------------------------------

                if (
                    phase == 'val'
                    and epoch_acc > best_acc
                ):

                    best_acc = epoch_acc

                    torch.save(
                        model.state_dict(),
                        best_model_params_path
                    )

            print()

        # --------------------------------------------------------
        # Fin de l'entraînement
        # --------------------------------------------------------

        time_elapsed = time.time() - since

        print(
            f'Training complete in '
            f'{time_elapsed // 60:.0f}m '
            f'{time_elapsed % 60:.0f}s'
        )

        print(
            f'Best val Acc: {best_acc:.4f}'
        )

        # --------------------------------------------------------
        # Chargement du meilleur modèle
        # --------------------------------------------------------

        model.load_state_dict(
            torch.load(
                best_model_params_path,
                weights_only=True
            )
        )

    return model


# ============================================================
# Visualisation des prédictions
# ============================================================

def visualize_model(model, num_images=6):

    was_training = model.training

    model.eval()

    images_so_far = 0

    fig = plt.figure()

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(
            dataloaders['val']
        ):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(
                outputs,
                1
            )

            for j in range(inputs.size()[0]):

                images_so_far += 1

                ax = plt.subplot(
                    num_images // 2,
                    2,
                    images_so_far
                )

                ax.axis('off')

                ax.set_title(
                    f'predicted: '
                    f'{class_names[preds[j].item()]}'
                )

                imshow(
                    inputs.cpu().data[j]
                )

                if images_so_far == num_images:

                    model.train(
                        mode=was_training
                    )

                    return

        model.train(
            mode=was_training
        )


# ============================================================
# 1. FINETUNING DU RESNET18
# ============================================================

print("\n")
print("=" * 60)
print("FINETUNING DU RESNET18")
print("=" * 60)


# Chargement du modèle pré-entraîné
model_ft = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)


# Nombre de neurones de la dernière couche
num_ftrs = model_ft.fc.in_features


# Remplacement de la dernière couche
# 2 classes : ants et bees
model_ft.fc = nn.Linear(
    num_ftrs,
    2
)


# Envoi du modèle vers MPS
model_ft = model_ft.to(device)


# Fonction de perte
criterion = nn.CrossEntropyLoss()


# Optimiseur
optimizer_ft = optim.SGD(
    model_ft.parameters(),
    lr=0.001,
    momentum=0.9
)


# Scheduler
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_ft,
    step_size=7,
    gamma=0.1
)


# ============================================================
# Entraînement du modèle Fine-Tuning
# ============================================================

model_ft = train_model(
    model_ft,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    num_epochs=25
)


# Visualisation
visualize_model(model_ft)


# ============================================================
# 2. RESNET18 COMME FEATURE EXTRACTOR
# ============================================================

print("\n")
print("=" * 60)
print("RESNET18 COMME FEATURE EXTRACTOR")
print("=" * 60)


# Chargement du ResNet18 pré-entraîné
model_conv = torchvision.models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)


# Gel de toutes les couches
for param in model_conv.parameters():

    param.requires_grad = False


# Nombre de neurones de la dernière couche
num_ftrs = model_conv.fc.in_features


# Nouvelle couche finale
model_conv.fc = nn.Linear(
    num_ftrs,
    2
)


# Envoi vers MPS
model_conv = model_conv.to(device)


# Fonction de perte
criterion = nn.CrossEntropyLoss()


# Seule la nouvelle couche finale est optimisée
optimizer_conv = optim.SGD(
    model_conv.fc.parameters(),
    lr=0.001,
    momentum=0.9
)


# Scheduler
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_conv,
    step_size=7,
    gamma=0.1
)


# ============================================================
# Entraînement Feature Extractor
# ============================================================

model_conv = train_model(
    model_conv,
    criterion,
    optimizer_conv,
    exp_lr_scheduler,
    num_epochs=25
)


# Visualisation
visualize_model(model_conv)


# ============================================================
# Affichage des graphiques
# ============================================================

plt.ioff()
plt.show()


# ============================================================
# Prédiction sur une image personnelle
# ============================================================

def visualize_model_predictions(
    model,
    img_path
):

    was_training = model.training

    model.eval()

    # Ouverture de l'image
    img = Image.open(img_path)

    # Transformation de l'image
    img = data_transforms['val'](img)

    # Ajout de la dimension batch
    img = img.unsqueeze(0)

    # Envoi vers MPS
    img = img.to(device)

    # Pas de calcul des gradients
    with torch.no_grad():

        outputs = model(img)

        _, preds = torch.max(
            outputs,
            1
        )

        ax = plt.subplot(2, 2, 1)

        ax.axis('off')

        ax.set_title(
            f'Predicted: '
            f'{class_names[preds[0]]}'
        )

        imshow(
            img.cpu().data[0]
        )

        model.train(
            mode=was_training
        )


# ============================================================
# Test sur une image de l'ensemble de validation
# ============================================================

visualize_model_predictions(
    model_conv,
    img_path=data_dir / 'val' / 'bees' / '2501530886_e20952b97d.jpg'
)


plt.ioff()
plt.show()
print("Done:")