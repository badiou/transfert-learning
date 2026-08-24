# Classification d'images : fourmis et abeilles

Ce projet applique le **transfer learning** avec PyTorch et Torchvision pour
classer des images de fourmis (*ants*) et d'abeilles (*bees*). Le script
principal est [transfert-learning.py](transfert-learning.py).

## Exemples d'images

| Fourmi | Abeille |
| --- | --- |
| ![Image d'une fourmi](images/fourmi.jpg) | ![Image d'une abeille](images/abeille.jpg) |

## Fonctionnement

Le script :

1. télécharge et décompresse automatiquement le dataset Hymenoptera s'il est absent ;
2. charge les ensembles `train` et `val` avec `ImageFolder` ;
3. transforme les images pour ResNet-18 : recadrage en `224 x 224`, augmentation
   aléatoire pour l'entraînement et normalisation ImageNet pour les deux ensembles ;
4. entraîne deux modèles ResNet-18 pré-entraînés sur ImageNet :
   - **fine-tuning** : toutes les couches sont entraînées ;
   - **feature extractor** : les couches pré-entraînées sont gelées et seule
     la couche finale est entraînée ;
5. affiche la `Loss` et l'accuracy (`Acc`) pour l'entraînement et la validation ;
6. affiche des prédictions sur des images de validation et sur une image de test.

## Prérequis

- Python 3.9 ou version ultérieure ;
- un environnement virtuel Python ;
- les dépendances de [requirements.txt](requirements.txt).

Le script utilise l'accélérateur disponible : CUDA, MPS sur les Mac Apple
Silicon, ou CPU. Le dataset et les poids pré-entraînés de ResNet-18 sont
téléchargés au premier lancement.

## Installation

Depuis le dossier du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

L'environnement `.venv` existe déjà ? Il suffit de l'activer dans chaque
nouvelle session de terminal :

```bash
source .venv/bin/activate
```

Dans VS Code, sélectionnez l'interpréteur `.venv/bin/python` avec la commande
**Python: Select Interpreter**.

## Exécution

```bash
source .venv/bin/activate
python transfert-learning.py
```

Le premier lancement nécessite une connexion Internet. Le script place le
dataset à côté de lui et ignore le téléchargement si `hymenoptera_data/`
existe déjà.

```text
transfert-learning/
├── transfert-learning.py
├── requirements.txt
├── README.md
├── images/
│   ├── fourmi.jpg
│   └── abeille.jpg
├── hymenoptera_data/
│   ├── train/
│   │   ├── ants/
│   │   └── bees/
│   └── val/
│       ├── ants/
│       └── bees/
└── .venv/
```

Le dataset est téléchargé depuis :
`https://download.pytorch.org/tutorial/hymenoptera_data.zip`.

## Paramètres principaux

Les paramètres sont définis directement dans
[transfert-learning.py](transfert-learning.py) :

| Paramètre | Valeur | Rôle |
| --- | ---: | --- |
| Taille des lots | `4` | Images traitées par lot |
| Nombre d'époques | `25` | Durée de chaque entraînement |
| Taux d'apprentissage | `0.001` | Pas de l'optimiseur SGD |
| Momentum | `0.9` | Paramètre de SGD |
| Scheduler | `StepLR` | Réduction tous les `7` époques par `0.1` |
| Workers | `0` | Compatible avec macOS et MPS |

Le meilleur état de chaque modèle est conservé temporairement selon
l'accuracy de validation, puis rechargé à la fin de l'entraînement. Aucun
modèle n'est enregistré dans un fichier permanent.

## Résultats d'entraînement

Résultats obtenus après `25` époques sur le dataset de validation :

| Méthode | Meilleure accuracy de validation | Durée |
| --- | ---: | ---: |
| Fine-tuning de ResNet-18 | **94,77 %** | 2 min 33 s |
| ResNet-18 comme feature extractor | **94,77 %** | 1 min 12 s |

Les deux méthodes obtiennent le même meilleur score de validation. Le mode
**feature extractor** est toutefois plus rapide, car les couches pré-entraînées
restent gelées et seule la couche finale est optimisée.

Pour le fine-tuning, la meilleure accuracy est atteinte aux époques `17` et
`19`. Pour le feature extractor, elle est atteinte à plusieurs époques, dont
`10`, `12`, `13`, `15`, `18`, `19`, `20` et `24`.

## Tester une autre image

À la fin du script, l'image de validation utilisée est définie ici :

```python
visualize_model_predictions(
    model_conv,
    img_path=data_dir / 'val' / 'bees' / '2501530886_e20952b97d.jpg'
)
```

Pour tester une autre image, remplacez ce chemin par celui d'une image lisible
par Pillow. La prédiction est affichée après les deux entraînements.

## Fichiers ignorés

[.gitignore](.gitignore) exclut notamment l'environnement virtuel, le dataset,
les archives ZIP, les poids de modèles et les caches Python.

---

# Image Classification: Ants and Bees

This project uses **transfer learning** with PyTorch and Torchvision to
classify images of ants and bees. The main script is
[transfert-learning.py](transfert-learning.py).

## Image Examples

| Ant | Bee |
| --- | --- |
| ![An ant](images/fourmi.jpg) | ![A bee](images/abeille.jpg) |

## How It Works

The script:

1. automatically downloads and extracts the Hymenoptera dataset if it is missing;
2. loads the `train` and `val` sets with `ImageFolder`;
3. transforms the images for ResNet-18: `224 x 224` crops, random augmentation
   for training, and ImageNet normalization for both sets;
4. trains two ImageNet-pre-trained ResNet-18 models:
   - **fine-tuning**: all layers are trained;
   - **feature extractor**: pre-trained layers are frozen and only the final
     layer is trained;
5. displays `Loss` and accuracy (`Acc`) for training and validation;
6. displays predictions for validation images and one test image.

## Requirements

- Python 3.9 or later;
- a Python virtual environment;
- the dependencies listed in [requirements.txt](requirements.txt).

The script uses the available accelerator: CUDA, MPS on Apple Silicon Macs,
or CPU. The dataset and ResNet-18 pre-trained weights are downloaded on the
first run.

## Installation

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If `.venv` already exists, activate it at the start of each new terminal
session:

```bash
source .venv/bin/activate
```

In VS Code, select `.venv/bin/python` with **Python: Select Interpreter**.

## Running the Script

```bash
source .venv/bin/activate
python transfert-learning.py
```

The first run requires an Internet connection. The script stores the dataset
next to itself and skips the download if `hymenoptera_data/` already exists.

The dataset is downloaded from:
`https://download.pytorch.org/tutorial/hymenoptera_data.zip`.

## Main Parameters

The parameters are defined directly in
[transfert-learning.py](transfert-learning.py):

| Parameter | Value | Purpose |
| --- | ---: | --- |
| Batch size | `4` | Images processed in each batch |
| Number of epochs | `25` | Duration of each training run |
| Learning rate | `0.001` | SGD optimizer step size |
| Momentum | `0.9` | SGD parameter |
| Scheduler | `StepLR` | Reduces the rate every `7` epochs by `0.1` |
| Workers | `0` | Compatible with macOS and MPS |

The best state of each model is temporarily saved according to validation
accuracy, then reloaded at the end of training. No model is saved permanently.

## Training Results

Results obtained after `25` epochs on the validation dataset:

| Method | Best validation accuracy | Duration |
| --- | ---: | ---: |
| ResNet-18 fine-tuning | **94.77%** | 2 min 33 sec |
| ResNet-18 as a feature extractor | **94.77%** | 1 min 12 sec |

Both methods reached the same best validation score. The **feature extractor**
approach was faster because the pre-trained layers remained frozen and only the
final layer was optimized.

For fine-tuning, the best accuracy was reached at epochs `17` and `19`. For the
feature extractor, it was reached at several epochs, including `10`, `12`,
`13`, `15`, `18`, `19`, `20`, and `24`.

## Testing Another Image

At the end of the script, the validation image is selected here:

```python
visualize_model_predictions(
    model_conv,
    img_path=data_dir / 'val' / 'bees' / '2501530886_e20952b97d.jpg'
)
```

To test another image, replace this path with the path to an image readable by
Pillow. The prediction is displayed after both training runs.

## Ignored Files

[.gitignore](.gitignore) excludes the virtual environment, dataset, ZIP
archives, model weights, and Python caches.