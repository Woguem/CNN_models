"""
@author: Dr Yen Fred WOGUEM 

@description: This script trains a CNN model to predict dislocation coordinates and their probability of presence

"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchvision
import os
import json
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
import joblib 


start_time = datetime.now()  # Démarrer le chronomètre



# Supposons que `device` est déjà défini, soit 'cuda' soit 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)



# ==============================
# 1️ Hyperparamètres
# ==============================
batch_size = 32 # Nombre d'images par lot
learning_rate = 0.00001
num_epochs = 2#200


# ==============================
# 2️ Chargement des données 
# ==============================

# Charger le fichier JSON
json_file = r"C:\Users\p09276\Post_doc_Yen_Fred\Projet_Machine_Learning_Julien\GB_Cu_001_Generation\Generated_Images\Grain_Boundary.json"  # Chemin vers le fichier JSON contenant les labels



#Construction du dataset
class CustomDataset(Dataset):
    def __init__(self, root, json_file, transform=None):
        self.root = root  # Répertoire contenant les images
        self.json_file = json_file  # Chemin vers le fichier JSON
        self.transform = transform  # Transformations à appliquer
        self.labels_data = self.load_json_data()  # Charger les données JSON
        self.image_names = self.get_image_files()  # Liste des noms d'images (seulement celles avec une extension valide)
        
        # Créer un scaler pour transformer les labels dans la plage [0, 1]
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Accumuler toutes les positions
        all_positions = []
        for image_name in self.image_names:
            dislocations = self.labels_data[image_name]["dislocations"]
            positions = []
            for dislocation in dislocations:
                if "x" in dislocation and "y" in dislocation:
                    positions.append([dislocation["x"], dislocation["y"]])
            all_positions.extend(positions)
        
        # Convertir en numpy array et appliquer fit
        all_positions = torch.tensor(all_positions, dtype=torch.float32).numpy()  # Convertir en numpy array
        self.scaler.fit(all_positions)  # Fit sur tout le dataset

        # Sauvegarde du scaler après l'avoir fit
        scaler_path = os.path.join(self.root, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)

    def get_image_files(self):
        """Retourne une liste des images valides (avec extension .png, .jpg, etc.)"""
        valid_extensions = ('.png', '.jpg', '.jpeg')   # Extensions valides
        image_files = [f for f in os.listdir(self.root) if f.lower().endswith(valid_extensions)]
        return image_files    
    
    def load_json_data(self):
        """Charge le fichier JSON et retourne les données"""
        with open(self.json_file, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        """Retourne le nombre total d'images"""
        return len(self.image_names)
    
    def __getitem__(self, idx):
        """Retourne une image et ses labels"""
        # Récupérer le nom de l'image
        image_name = self.image_names[idx]
        
        
        # Charger l'image
        img_path = os.path.join(self.root, image_name)
        image = Image.open(img_path).convert("RGB")
        
        # Obtenir les dislocations et leurs labels (positions x, y et p1, p0)
        dislocations = self.labels_data[image_name]["dislocations"]
        
        # Préparer les labels (positions + probabilités)
        positions = []
        probabilities = []
        
            
        for dislocation in dislocations:
            # On filtre les dislocations ayant des coordonnées valides
            if "x" in dislocation and "y" in dislocation :
                positions.append([dislocation["x"], dislocation["y"]])
                probabilities.append([dislocation["p1"], dislocation["p0"]])
        
        # Transformer les psoitions dans l'interval (0,1) et convertir en tensors
        positions =  np.array(positions, dtype=np.float32)     #torch.tensor(positions, dtype=torch.float32)
        positions = self.scaler.transform(positions)
        positions = torch.tensor(positions, dtype=torch.float32) 
        probabilities = torch.tensor(probabilities, dtype=torch.float32)

            
        
        
        # Appliquer les transformations (si fournies)
        if self.transform:
            image = self.transform(image)
        
        return image, positions, probabilities


#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), Normalisation RGB pour avoir des valeur entre (-1,1)

transform = transforms.Compose([transforms.Resize((20, 20)),
    transforms.ToTensor(),    # Convertit l'image en tenseur PyTorch et les pixels sont désormais entre (0,1)
    ])


# Créer le dataset

root = r"C:\Users\p09276\Post_doc_Yen_Fred\Projet_Machine_Learning_Julien\GB_Cu_001_Generation\Generated_Images"  # Chemin vers le dossier contenant les images


# Vérification des chemins
if not os.path.exists(root):
    print(f"Le répertoire des images {root} n'existe pas.")
else:
    print(f"Le répertoire des images existe : {root}")

if not os.path.exists(json_file):
    print(f"Le fichier JSON {json_file} n'existe pas.")
else:
    print(f"Le fichier JSON existe : {json_file}")



dataset = CustomDataset(root=root, json_file=json_file, transform=transform)




def custom_collate(batch):
    """
    Fonction pour gérer les tailles variables dans un batch de données.
    Elle regroupe les images normalement, et applique du padding aux positions et probabilités.
    """
    # Empiler les images normalement (assurez-vous que les images sont déjà sous forme de tenseurs)
    images = torch.stack([item[0] for item in batch])  
    
    # Garder les positions et probabilités comme des listes de listes de tenseurs
    positions = [item[1] for item in batch]
    probabilities = [item[2] for item in batch]
    
    # Appliquer le padding sur les positions et les probabilités
    # Le padding est effectué pour avoir des séquences de même longueur (batch_first=True)
    padded_positions = pad_sequence(positions, batch_first=True, padding_value=0.0)  # Padding sur positions
    padded_probabilities = pad_sequence(probabilities, batch_first=True, padding_value=0.0)  # Padding sur probabilités

    

    # Créer un masque pour ignorer les éléments paddés (1 pour les vrais, 0 pour les paddés)
    positions_mask = (padded_positions.sum(dim=-1) != 0).float()  # Masque pour les positions non-paddées
    probabilities_mask = (padded_probabilities.sum(dim=-1) != 0).float()  # Masque pour les probabilités non-paddées


    return images, padded_positions, padded_probabilities, positions_mask, probabilities_mask



train_size = int(0.6 * len(dataset))  # 60% pour l'entraînement
val_size = int(0.2 * len(dataset))   # 20% pour la validation
test_size = len(dataset) - train_size - val_size  # Le reste pour le test

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=0, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=0, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate, num_workers=0, shuffle=False)


print(len(train_loader))
print(len(val_loader))
print(len(test_loader))  



'''
# Affichage des résultats

print(f"Nombre d'images : {len(dataset)}") # Taille du dataset et de l'image

#image = dataset[0]  # Cela te donnera la première image après transformation.
#print(f"Image à l'index 0 : {image}")

# Récupérer le batch
batch = next(iter(train_loader))

print(batch)

# Décomposer le batch
images = batch[0]  # Les images
padded_positions = batch[1]  # Les positions après padding
padded_probabilities = batch[2]  # Les probabilités après padding
positions_mask = batch[3]  # Le masque pour les positions
probabilities_mask = batch[4]  # Le masque pour les probabilités

# Afficher les shapes des différents éléments
print("Images:", images.shape)  # (Batch_size, Channels, Height, Width)
print("Padded positions:", padded_positions.shape)  # (Batch_size, max_dislocations, 2) - coordonnées x, y
print("Padded probabilities:", padded_probabilities.shape)  # (Batch_size, max_dislocations, 2) - p1, p0
print("Positions mask:", positions_mask.shape)  # (Batch_size, max_dislocations)
print("Probabilities mask:", probabilities_mask.shape)  # (Batch_size, max_dislocations)





# Convertir les images de [-1,1] à [0,1] pour les afficher correctement
images = images / 2 + 0.5  # Dé-normalisation

# Afficher quelques images
fig, axes = plt.subplots(2, 1, figsize=(10, 5)) #4, 8
axes = axes.flatten()
for img, ax in zip(images[:32], axes):
    img = img.numpy().transpose((1, 2, 0))  # Convertir au format (H, W, C)
    ax.imshow(img)
    ax.axis("off")

plt.show()'''





# ==============================
# 3️ Définition du Modèle CNN
# ==============================
class CNN(nn.Module):
    def __init__(self, num_classes, num_coordinates):
        super(CNN, self).__init__()
        
        
        self.num_classes = num_classes  # Nombre de classes de probabilités
        self. num_coordinates =  num_coordinates  # Nombre de coordonnées

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, padding=1)
        
        # Activation layer
        self.relu = nn.ReLU()
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        
    def forward(self, x, padded_positions):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)  # Aplatir
        
        # Fully connected layers
        flattened_size = 512 * 3 * 3  # Taille après les convolutions

        # Mise à jour de max_dislocations_in_batch pendant le passage avant
        self.max_dislocations_in_batch = padded_positions.size(1)  # Calcul dynamique basé sur chaque batch dans la boucle d'entrainement
        
        #print(f"max_dislocations_in_batch: {self.max_dislocations_in_batch}")

        # Sortie classification (num_classes = 2 (p0, p1), pour chaque dislocation) 
        self.classification_head = nn.Linear(flattened_size, self.max_dislocations_in_batch * num_classes)  

        # Sortie régression (num_coordinates = 2 (x, y), pour chaque dislocation)
        self.regression_head = nn.Linear(flattened_size, self.max_dislocations_in_batch * num_coordinates)  # output = torch.tensor([[x1_1, y1_1, x2_1, y2_1, x3_1, y3_1]])  # 1 image, 3 dislocations, 2 coordonnées (x, y) par dislocation

        # Sortie classification (Batch, max_dislocations, 2)
        classification_logits = self.classification_head(x).view(-1, self.max_dislocations_in_batch, self.num_classes) #  Utilisé pour le loss  
        
        #classification_probs = F.softmax(classification_logits, dim=-1)  # Converts to probabilities (just for display), but is not used for loss because nn.CrossEntropyLoss already uses softmax (you must use the one with logit).

        # Sortie régression (Batch, max_dislocations, 2)
        regression_output = self.regression_head(x).view(-1, self.max_dislocations_in_batch, self.num_coordinates) # output = torch.tensor([[x1_1, y1_1], [x2_1, y2_1], [x3_1, y3_1]])  # 1 image, 3 dislocations, 2 coordonnées (x, y) par dislocation
  

        return classification_logits, regression_output

# ==============================
# 4️ Initialisation du modèle
# ==============================

num_classes = 2 # Nombre de classes (p1, p0)
num_coordinates = 2 # Nombre de coordonnées (x, y)

model = CNN(num_classes = num_classes, num_coordinates = num_coordinates).to(device)

criterion_classification = nn.CrossEntropyLoss(reduction='none')
criterion_regression = nn.SmoothL1Loss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Scheduler : diminue le taux d'apprentissage de 10% toutes les 10 époques
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)



# ==============================
# 5️ Boucle d'entraînement
# ==============================


def train_model(model, train_loader, val_loader, criterion_classification, criterion_regression, optimizer, device, num_epochs, scheduler=None):
    """
    Fonction d'entraînement du modèle avec validation toutes les 2 époques.

    Args:
        model: Le modèle à entraîner
        train_loader: DataLoader pour l'entraînement
        val_loader: DataLoader pour la validation
        criterion_classification: Fonction de perte pour la classification
        criterion_regression: Fonction de perte pour la régression
        optimizer: Optimiseur
        device: "cuda" ou "cpu"
        num_epochs: Nombre total d'époques
        scheduler: (Optionnel) Scheduler pour ajuster le learning rate

    Returns:
        Dictionnaire contenant les métriques d'entraînement et de validation
    """

    # Listes pour stocker les métriques
    metrics = {
        "Epochs": [],
        "train_loss_classification": [],
        "train_loss_regression": [],
        "train_loss_total": [],
        "train_accuracy_classification": [],
        "train_accuracy_regression": [],
        "train_accuracy_total": [],
        "val_loss_classification": [],
        "val_loss_regression": [],
        "val_loss_total": [],
        "val_accuracy_classification": [],
        "val_accuracy_regression": [],
        "val_accuracy_total": [],
    }

    for epoch in range(num_epochs):
        model.train()  # Mode entraînement
        
        running_loss_classification = 0.0
        running_loss_regression = 0.0
        running_loss_total = 0.0
        correct_classification = 0
        correct_regression = 0
        total = 0

        for images, padded_positions, padded_probabilities, positions_mask, probabilities_mask in train_loader:
            # Envoi des données sur GPU/CPU
            images = images.to(device)
            padded_positions = padded_positions.to(device)
            padded_probabilities = padded_probabilities.to(device)
            positions_mask = positions_mask.to(device)
            probabilities_mask = probabilities_mask.to(device)

            optimizer.zero_grad()  # Reset des gradients

            # Passage avant
            classification_logits, regression_output = model(images, padded_positions)

            # Aplatir les logits : [batch_size * max_dislocations, num_classes]
            classification_logits = classification_logits.view(-1, 2)  
            
            # Transformer padded_probabilities en indices de classes et aplatir les cibles : [batch_size * max_dislocations]
            target_classes = padded_probabilities.argmax(dim=-1).view(-1)

            # Calcul des pertes
            loss_classification = criterion_classification(classification_logits, target_classes)
            loss_regression = criterion_regression(regression_output, padded_positions)

            # Appliquer les masques
            loss_regression *= positions_mask.unsqueeze(-1)
            loss_classification *= probabilities_mask.view(-1)

            # Moyenne des pertes en fonction des éléments valides
            loss_regression = loss_regression.sum() / (2*positions_mask.sum()) # 2 car il y a deux coordonnées (x,y)
            loss_classification = loss_classification.sum() / probabilities_mask.sum()

            # Somme des pertes
            total_loss = loss_classification + loss_regression

            # Rétropropagation et mise à jour des poids
            total_loss.backward()
            optimizer.step()

            # Mise à jour des métriques
            running_loss_classification += loss_classification.item()
            running_loss_regression += loss_regression.item()
            running_loss_total += total_loss.item()

            # Calcul de la précision classification
            _, predicted_classes = torch.max(classification_logits, 1)
            correct_classification += (predicted_classes == target_classes).sum().item()

            # Calcul de la précision régression (ex: distance < seuil)
            correct_regression += (torch.abs(regression_output - padded_positions) < 0.1).sum().item()

            total += target_classes.size(0)

        if (epoch + 1) % 2 == 0:
            # Moyenne des pertes et précisions sur l'ensemble du train_loader
            metrics['Epochs'].append(epoch + 1)
            metrics["train_loss_classification"].append(running_loss_classification / len(train_loader))
            metrics["train_loss_regression"].append(running_loss_regression / len(train_loader))
            metrics["train_loss_total"].append(running_loss_total / len(train_loader))
            metrics["train_accuracy_classification"].append(100 * correct_classification / total)
            metrics["train_accuracy_regression"].append(100 * correct_regression / (total * regression_output.shape[-1]))  # Par feature
            metrics["train_accuracy_total"].append((metrics["train_accuracy_classification"][-1] + metrics["train_accuracy_regression"][-1]) / 2)

            # Affichage des stats de l'entraînement
            print(f"Époque {epoch + 1}/{num_epochs} \n "
                f"  ###### TRAIN ##### \n"
                f"Perte classification: {metrics['train_loss_classification'][-1]:.4f} \n "
                f"Perte régression: {metrics['train_loss_regression'][-1]:.4f} \n "
                f"Perte totale: {metrics['train_loss_total'][-1]:.4f} \n "
                f"Précision classification: {metrics['train_accuracy_classification'][-1]:.2f}% \n "
                f"Précision régression: {metrics['train_accuracy_regression'][-1]:.2f}% \n "
                f"Précision totale: {metrics['train_accuracy_total'][-1]:.2f}%")

        # Appliquer le scheduler si défini
        if scheduler:
            scheduler.step()

        # ==============================
        # 6️ Validation toutes les 2 époques
        # ==============================

        if (epoch + 1) % 2 == 0:
            val_metrics = validate_model(model, val_loader, criterion_classification, criterion_regression, device)
            for key in val_metrics:
                metrics[key].append(val_metrics[key])

            print(f"  ###### VALIDATION ##### \n"
                  f"Perte classification: {metrics['val_loss_classification'][-1]:.4f} \n"
                  f"Perte régression: {metrics['val_loss_regression'][-1]:.4f} \n"
                  f"Perte totale: {metrics['val_loss_total'][-1]:.4f} \n"
                  f"Précision classification: {metrics['val_accuracy_classification'][-1]:.2f}% \n"
                  f"Précision régression: {metrics['val_accuracy_regression'][-1]:.2f}% \n"
                  f"Précision totale: {metrics['val_accuracy_total'][-1]:.2f}%")

    return metrics



def validate_model(model, val_loader, criterion_classification, criterion_regression, device):
    """
    Fonction de validation du modèle.

    Returns:
        Dictionnaire contenant les pertes et précisions de validation
    """

    model.eval()  # Mode évaluation
    val_loss_classification = 0.0
    val_loss_regression = 0.0
    val_loss_total = 0.0
    correct_classification = 0
    correct_regression = 0
    total = 0

    with torch.no_grad():
        for images, padded_positions, padded_probabilities, positions_mask, probabilities_mask in val_loader:
            images = images.to(device)
            padded_positions = padded_positions.to(device)
            padded_probabilities = padded_probabilities.to(device)
            positions_mask = positions_mask.to(device)
            probabilities_mask = probabilities_mask.to(device)

            classification_logits, regression_output = model(images, padded_positions)

            classification_logits = classification_logits.view(-1, 2)
            target_classes = padded_probabilities.argmax(dim=-1).view(-1)

            loss_classification = criterion_classification(classification_logits, target_classes)
            loss_regression = criterion_regression(regression_output, padded_positions)

            loss_regression *= positions_mask.unsqueeze(-1)
            loss_classification *= probabilities_mask.view(-1)

            loss_regression = loss_regression.sum() / (2*positions_mask.sum())
            loss_classification = loss_classification.sum() / probabilities_mask.sum()

            total_loss = loss_classification + loss_regression

            val_loss_classification += loss_classification.item()
            val_loss_regression += loss_regression.item()
            val_loss_total += total_loss.item()

            _, predicted_classes = torch.max(classification_logits, 1)
            correct_classification += (predicted_classes == target_classes).sum().item()
            correct_regression += (torch.abs(regression_output - padded_positions) < 0.1).sum().item()

            total += target_classes.size(0)

    return {
        "val_loss_classification": val_loss_classification / len(val_loader),
        "val_loss_regression": val_loss_regression / len(val_loader),
        "val_loss_total": val_loss_total / len(val_loader),
        "val_accuracy_classification": 100 * correct_classification / total,
        "val_accuracy_regression": 100 * correct_regression / (total * regression_output.shape[-1]),
        "val_accuracy_total": (100 * correct_classification / total + 100 * correct_regression / (total * regression_output.shape[-1])) / 2
    }




# Appel de la fonction d'entraînement
metrics  = train_model( model, train_loader, val_loader, 
    criterion_classification, criterion_regression, optimizer, device, num_epochs, scheduler = None
)


#for key in metrics:
 #   print(f"{key}: {len(metrics[key])}")

    


def plot_loss_accuracy(metrics, save_dir):
    """
    Fonction pour créer et enregistrer un graphique montrant l'évolution des pertes et précisions
    pour l'entraînement et la validation.

    Args:
        metrics (dict): Dictionnaire contenant les métriques d'entraînement et de validation.
        save_dir (str): Répertoire où sauvegarder l'image du graphique.

    Returns:
        str: Chemin du fichier enregistré.
    """
    
    # Création des données pour l'entraînement
    df_train = pd.DataFrame({
        "Epoch": metrics["Epochs"]* 6,  # Répéter les epochs pour les trois types de loss/accuracy
        "Loss_Accuracy": metrics["train_loss_classification"] + metrics["train_loss_regression"] + metrics["train_loss_total"] +
                        metrics["train_accuracy_classification"] + metrics["train_accuracy_regression"] + metrics["train_accuracy_total"],
        "Type": (["Loss_classification"] * len(metrics["train_loss_classification"]) + 
                 ["Loss_regression"] * len(metrics["train_loss_regression"]) + 
                 ["Loss_total"] * len(metrics["train_loss_total"]) + 
                 ["Accuracy_classification"] * len(metrics["train_accuracy_classification"]) + 
                 ["Accuracy_regression"] * len(metrics["train_accuracy_regression"]) + 
                 ["Accuracy_total"] * len(metrics["train_accuracy_total"])),
        "Dataset": ["Train"] * len(metrics["train_loss_classification"])*6
    })

    # Création des données pour la validation
    df_val = pd.DataFrame({
        "Epoch": metrics["Epochs"]* 6,
        "Loss_Accuracy": metrics["val_loss_classification"] + metrics["val_loss_regression"] + metrics["val_loss_total"] +
                         metrics["val_accuracy_classification"] + metrics["val_accuracy_regression"] + metrics["val_accuracy_total"],
        "Type": (["Loss_classification"] * len(metrics["val_loss_classification"]) + 
                 ["Loss_regression"] * len(metrics["val_loss_regression"]) + 
                 ["Loss_total"] * len(metrics["val_loss_total"]) + 
                 ["Accuracy_classification"] * len(metrics["val_accuracy_classification"]) + 
                 ["Accuracy_regression"] * len(metrics["val_accuracy_regression"]) + 
                 ["Accuracy_total"] * len(metrics["val_accuracy_total"])),
        "Dataset": ["Validation"] * len(metrics["val_loss_classification"])*6
    })


    # Combiner les DataFrames
    df = pd.concat([df_train, df_val], ignore_index=True)

    # Configuration de Seaborn
    sns.set_theme(style="darkgrid")

    # Création de la figure
    plt.figure(figsize=(18, 10))

    # Graphiques pour l'entraînement
    for i, loss_type in enumerate(["Loss_classification", "Loss_regression", "Loss_total"]):
        plt.subplot(2, 6, i + 1)
        sns.lineplot(x="Epoch", y="Loss_Accuracy", hue="Type", data=df[(df["Dataset"] == "Train") & (df["Type"] == loss_type)], marker="o")
        plt.title(f"Train - {loss_type.replace('_', ' ').title()} ")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

    for i, accuracy_type in enumerate(["Accuracy_classification", "Accuracy_regression", "Accuracy_total"]):
        plt.subplot(2, 6, i + 4)
        sns.lineplot(x="Epoch", y="Loss_Accuracy", hue="Type", data=df[(df["Dataset"] == "Train") & (df["Type"] == accuracy_type)], marker="o")
        plt.title(f"Train - {accuracy_type.replace('_', ' ').title()} ")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

    # Graphiques pour la validation
    for i, loss_type in enumerate(["Loss_classification", "Loss_regression", "Loss_total"]):
        plt.subplot(2, 6, i + 7)
        sns.lineplot(x="Epoch", y="Loss_Accuracy", hue="Type", data=df[(df["Dataset"] == "Validation") & (df["Type"] == loss_type)], marker="o")
        plt.title(f"Validation - {loss_type.replace('_', ' ').title()} ")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

    for i, accuracy_type in enumerate(["Accuracy_classification", "Accuracy_regression", "Accuracy_total"]):
        plt.subplot(2, 6, i + 10)
        sns.lineplot(x="Epoch", y="Loss_Accuracy", hue="Type", data=df[(df["Dataset"] == "Validation") & (df["Type"] == accuracy_type)], marker="o")
        plt.title(f"Validation - {accuracy_type.replace('_', ' ').title()} ")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
   

    # Ajustement de l'affichage
    plt.tight_layout()

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Sauvegarde du graphique
    Plot_file_path = os.path.join(save_dir, 'loss_accuracy_plot.png')
    plt.savefig(Plot_file_path)
    plt.close()

    return Plot_file_path

# Spécifie le répertoire où tu veux enregistrer l'image
save_dir = r"C:\Users\p09276\Post_doc_Yen_Fred\Projet_Machine_Learning_Julien\GB_Cu_001_Generation\Loss_Accuracy_Plots"  # Remplace par le chemin du répertoire de ton choix


Plot_file_path = plot_loss_accuracy(metrics, save_dir)

print(f"Graphique sauvegardé dans : {Plot_file_path}")



def test_model(model, test_loader, criterion_classification, criterion_regression, device, save_dir_prediction_true, scaler_path):
    model.eval()  # Passer en mode évaluation
    test_loss_classification = 0.0
    test_loss_regression = 0.0
    test_loss_total = 0.0
    correct_classification = 0
    correct_regression = 0
    total = 0
    
    predictions_true_list = []

    # Charger le scaler
    scaler = joblib.load(scaler_path)

    with torch.no_grad():  # Désactiver le calcul des gradients pendant le test
        for images, padded_positions, padded_probabilities, positions_mask, probabilities_mask in test_loader:
            
            images = images.to(device)
            padded_positions = padded_positions.to(device)
            padded_probabilities = padded_probabilities.to(device)
            positions_mask = positions_mask.to(device)
            probabilities_mask = probabilities_mask.to(device)
            
            # Propagation avant
            classification_logits, regression_output = model(images, padded_positions)

            # Probabilités
            classification_probs = F.softmax(classification_logits, dim=-1)

            # Caclul de la perte pour la classification
            classification_logits = classification_logits.view(-1, 2)
            target_classes = padded_probabilities.argmax(dim=-1).view(-1)
            loss_classification = criterion_classification(classification_logits, target_classes)

            # Calcul de la perte pour la régression
            loss_regression = criterion_regression(regression_output, padded_positions)

            # Appliquer les masques pour ignorer les valeurs padding
            loss_regression = loss_regression * positions_mask.unsqueeze(-1)
            loss_classification = loss_classification * probabilities_mask.view(-1)

            # Moyenne de la perte
            loss_regression = loss_regression.sum() / (2*positions_mask.sum())
            loss_classification = loss_classification.sum() / probabilities_mask.sum()

            # Perte totale
            total_loss = loss_classification + loss_regression


            test_loss_classification += loss_classification.item()
            test_loss_regression += loss_regression.item()
            test_loss_total += total_loss.item()

            _, predicted_classes = torch.max(classification_logits, 1)
            correct_classification += (predicted_classes == target_classes).sum().item()
            correct_regression += (torch.abs(regression_output - padded_positions) < 0.1).sum().item()

            total += target_classes.size(0)

            # Pour l'affichage des résultats dans un fihier JSON

            predictions_positions_sans_mask = regression_output*positions_mask.unsqueeze(-1) # Remove predicted padded positions
            predictions_probabilities_sans_mask = classification_probs*probabilities_mask.unsqueeze(-1) # Remove predicted padded probabilities

            # Conversion en numpy pour appliquer inverse_transform
            predicted_positions_np = predictions_positions_sans_mask.numpy()
            true_positions_np = padded_positions.numpy()

            # Inverse transform pour récupérer les vraies valeurs
            predicted_positions_real = np.array([scaler.inverse_transform(batch) for batch in predicted_positions_np])
            true_positions_real = np.array([scaler.inverse_transform(batch) for batch in true_positions_np])

            # Sauvegarde des résultats pour chaque élément du batch
            for i in range(len(true_positions_real)):
                predictions_true_list.append({
                    "Positions": {
                        "True":true_positions_real[i].tolist(),
                        "Predicted": predicted_positions_real[i].tolist()
                    },
                    "Probabilities": {
                        "True": padded_probabilities[i].tolist(),
                        "Predicted": predictions_probabilities_sans_mask[i].tolist()
                    }
                })


    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(save_dir_prediction_true):
        os.makedirs(save_dir_prediction_true)

    file_prediction_true = os.path.join(save_dir_prediction_true, 'Predictions_and_True_Positions.json')    
    # Sauvegarde des prédictions dans un fichier JSON
    with open(file_prediction_true, "w") as f:
        json.dump(predictions_true_list, f, indent=4)

    print(f"Prédictions et valeurs réelles sauvegardés dans {file_prediction_true}")
    
    
    # Affichage des résultats
    print(f"###### TEST ######\n")
    print(f"Perte de classification : {test_loss_classification / len(test_loader):.4f}\n")
    print(f"Perte de régression : {test_loss_regression / len(test_loader):.4f}\n")
    print(f"Perte totale : {test_loss_total / len(test_loader):.4f}\n")
    print(f"Précision classification : {100 * correct_classification / total:.2f}%\n")
    print(f"Précision régression : {100 * correct_regression / (total * regression_output.shape[-1]):.2f}%\n")
    print(f"Précision totale : {(100 * correct_classification / total + 100 * correct_regression / (total * regression_output.shape[-1])) / 2:.2f}%")

    return {
        "test_loss_classification": test_loss_classification / len(test_loader),
        "test_loss_regression": test_loss_regression / len(test_loader),
        "test_loss_total": total_loss / len(test_loader),
        "test_accuracy_classification": 100 * correct_classification / total,
        "test_accuracy_regression": 100 * correct_regression / (total * regression_output.shape[-1]),
        "test_accuracy_total": (100 * correct_classification / total + 100 * correct_regression / (total * regression_output.shape[-1])) / 2
    }


save_dir_prediction_true = r"C:\Users\p09276\Post_doc_Yen_Fred\Projet_Machine_Learning_Julien\GB_Cu_001_Generation\Predictions_And_True"  # Remplace par le chemin du répertoire de ton choix

scaler_path = os.path.join(root, "scaler.pkl")

# Appel de la fonction de test
test_metrics = test_model(model, test_loader, criterion_classification, criterion_regression, device, save_dir_prediction_true, scaler_path)









end_time = datetime.now()  # Fin du chronomètre
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")


















