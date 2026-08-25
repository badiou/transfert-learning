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
import random
from PIL import Image
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
import kagglehub
from famous_dataset_import import download_dataset

# Le device est compatible avec CUDA, MPS (macOS) et CPU.
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)
cudnn.benchmark = device.type == "cuda"

cudnn.benchmark = True
plt.ion()
NUM_EPOCHS = 5

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



project_dir = Path(__file__).resolve().parent
models_dir = project_dir / "models"
models_dir.mkdir(exist_ok=True)

DATASET_NAME = "muhammadhananasghar/5-famous-people-face-recognition"
DATASET_DIR_NAME = "famous_people_dataset"

download_dataset(
    dataset_name=DATASET_NAME,
    destination_dir=project_dir / DATASET_DIR_NAME
)

data_dir = project_dir / DATASET_DIR_NAME / "data"
split_dirs = {
    'train': 'train',
    'val': 'valid',
}

if not data_dir.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {data_dir}"
    )

if not data_dir.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {data_dir}"
    )

image_datasets = {
    split: datasets.ImageFolder(
        data_dir / split_dirs[split],
        data_transforms[split]
    )
    for split in ['train', 'val']
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
    title=[class_names[x.item()] for x in classes]
)


# ============================================================
# Fonction d'entraînement
# ============================================================

def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    model_path,
    num_epochs=25
):
    if model_path.exists():
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
                weights_only=True
            )
        )
        print(f"Modèle chargé depuis : {model_path}")
        return model

    since = time.time()
    torch.save(model.state_dict(), model_path)
    best_acc = 0.0

    # --------------------------------------------------------
    # Boucle sur les epochs
    # --------------------------------------------------------
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(
                    preds == labels.data
                ).item()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(
                f'{phase} Loss: {epoch_loss:.4f} '
                f'Acc: {epoch_acc:.4f}'
            )

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), model_path)

        print()

    time_elapsed = time.time() - since
    print(
        f'Training complete in '
        f'{time_elapsed // 60:.0f}m '
        f'{time_elapsed % 60:.0f}s'
    )
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
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
    5  # 5 classes : bill_gates, elon_musk, jeff_bezos, mark_zuckerberg, steve_jobs
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
    models_dir / "fine_tuning_resnet18.pt",
    num_epochs=NUM_EPOCHS
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
    5 
    # 5 c'est le nombre de classes en sortie : bill_gates, elon_musk, jeff_bezos, mark_zuckerberg, steve_jobs
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
    models_dir / "feature_extractor_resnet18.pt",
    num_epochs=NUM_EPOCHS
)


# Visualisation
visualize_model(model_conv)


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
            f'{class_names[preds[0].item()]}'
        )

        imshow(
            img.cpu().data[0]
        )

        model.train(
            mode=was_training
        )


def visualize_random_validation_images(model):

    was_training = model.training
    model.eval()

    validation_dir = data_dir / split_dirs['val']
    images_per_row = 2
    rows = int(np.ceil(len(class_names) / images_per_row))

    plt.figure(figsize=(10, 5 * rows))

    with torch.no_grad():

        for index, class_name in enumerate(class_names, start=1):

            class_dir = validation_dir / class_name
            image_paths = list(class_dir.glob('*.jpg'))

            if not image_paths:
                raise FileNotFoundError(
                    f"Aucune image JPEG trouvée dans {class_dir}"
                )

            image_path = random.choice(image_paths)
            image = Image.open(image_path).convert('RGB')
            tensor = data_transforms['val'](image).unsqueeze(0).to(device)

            outputs = model(tensor)
            prediction = torch.argmax(outputs, dim=1).item()

            ax = plt.subplot(rows, images_per_row, index)
            ax.axis('off')
            ax.set_title(
                f"Réel : {class_name}\n"
                f"Prédit : {class_names[prediction]}"
            )
            imshow(tensor.cpu()[0])

    model.train(mode=was_training)


# ============================================================
# Test sur une image de l'ensemble de validation
# ============================================================

visualize_random_validation_images(model_conv)


print("Entraînement et validation terminés.")
print("Fermez les fenêtres Matplotlib pour quitter le programme.")

plt.ioff()
plt.show()
plt.close('all')
print("Programme terminé.")



