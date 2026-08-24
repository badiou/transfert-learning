# Classification d'images : fourmis et abeilles

Ce projet applique le **transfer learning** avec PyTorch et Torchvision pour
classer des images de fourmis (*ants*) et d'abeilles (*bees*). Le script
principal est [transfert-learning.py](transfert-learning.py).

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