import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split, DataLoader
import random

# 1. Configurações
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
torch.manual_seed(42)

# 2. Transforms e Dados
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalização
])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Subconjunto de validação (10% do treino)
val_size = int(0.1 * len(train_data))
train_size = len(train_data) - val_size
train_subset, val_subset = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 3. Definição da CNN (Exemplo de arquitetura flexível)
class CNN(nn.Module):
    def __init__(self, conv_layers=2, filters=[64, 128], use_dropout=True, dropout_p=0.3):
        super(CNN, self).__init__()
        layers = []
        in_channels = 3
        for i in range(conv_layers):
            layers.append(nn.Conv2d(in_channels, filters[i], kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = filters[i]
        
        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=(filters[conv_layers - 1] * 8 * 8), out_features=200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.dropout = nn.Dropout(dropout_p) if use_dropout else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# 4. Treinamento
def train_model(model, optimizer, criterion, epochs=10, early_stop=True):
    model.to(device)
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        acc = 100. * correct / total
        print(f"Época {epoch+1}: Perda={running_loss:.4f} | Acurácia={acc:.2f}%")

        # Validação
        val_loss = evaluate_model(model, val_loader, criterion, report=False)
        if early_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping ativado.")
                    break

# 5. Avaliação
def evaluate_model(model, loader, criterion, report=True):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    acc = 100. * correct / total
    if report:
        print(f"Avaliação: Perda média={loss:.4f} | Acurácia={acc:.2f}%")
    return loss

# 6. Visualizar filtros da primeira camada
def plot_filters(model):
    with torch.no_grad():
        filters = model.conv[0].weight.cpu().clone()
        fig, axs = plt.subplots(1, 6, figsize=(15, 3))
        for i in range(6):
            img = filters[i].permute(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            axs[i].imshow(img)
            axs[i].axis('off')
        plt.suptitle("Filtros da primeira camada convolucional")
        plt.show()

# 7. Mostrar imagens com predições
def show_predictions(model, loader):
    model.eval()
    classes = train_data.classes
    data_iter = iter(loader)
    images, labels = next(data_iter)
    images = images.to(device)
    outputs = model(images)
    _, preds = outputs.max(1)
    
    images = images.cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    for i in range(6):
        img = (images[i] * 0.5) + 0.5  # unnormalize
        img = np.transpose(img, (1, 2, 0))
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(f"Pred: {classes[preds[i]]} | Real: {classes[labels[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 8. Execução de exemplo
if __name__ == "__main__":
    model = CNN(conv_layers=3, filters=[64, 128, 128], use_dropout=True, dropout_p=0.3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_model(model, optimizer, criterion, epochs=20, early_stop=True)
    print("\nAvaliação no conjunto de teste:")
    evaluate_model(model, test_loader, criterion)

    plot_filters(model)
    show_predictions(model, test_loader)