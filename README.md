# 🧠 Classificação de Imagens com CNNs - CIFAR-10 (Atividade 4 - Redes Neurais Artificiais)

Este projeto implementa redes neurais convolucionais (CNNs) utilizando **PyTorch** para classificação de imagens do dataset **CIFAR-10**, como parte da Atividade 4 da disciplina de Redes Neurais Artificiais (PPGCC - UNESP).

---

## 📚 Objetivo

Implementar e treinar diferentes arquiteturas de redes convolucionais para classificar imagens em 10 categorias do conjunto CIFAR-10, explorando:

- Arquiteturas com diferentes profundidades e número de filtros
- Uso de Dropout para reduzir overfitting
- Avaliação do desempenho com métricas de acurácia
- Visualização de filtros e predições

---

## 🛠️ Tecnologias Utilizadas

- Python 3.10+
- PyTorch
- Torchvision
- Matplotlib
- NumPy

---

## 📦 Instalação e Execução

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/seu-usuario/cnn-cifar10-atividade4.git
cd cnn-cifar10-atividade4
pip install -r requirements.txt
```

Execute o script principal:

```bash
python cnn_cifar10.py
```

---

## 🧪 Arquiteturas Testadas

| Modelo  | Camadas Convolucionais | Filtros                 | Dropout (FC) | Acurácia Validação | Acurácia Teste | Overfitting |
|---------|-------------------------|-------------------------|---------------|---------------------|----------------|-------------|
| CNN 1   | 2                       | [32, 64]                | 0.1           | 71.2%               | 70.1%          | Leve        |
| CNN 2   | 3                       | [64, 128, 128]          | 0.3           | 78.4%               | 77.0%          | Reduzido    |
| CNN 3   | 4                       | [64, 64, 128, 128]      | 0.3           | 79.1%               | 78.5%          | Reduzido    |
| CNN 4   | 3                       | [64, 128, 128]          | **sem Dropout** | 81.2%               | 74.8%          | Alto        |

---

## 🔍 Resultados e Conclusões

- A arquitetura **CNN 3** (4 camadas + Dropout) apresentou o melhor equilíbrio entre complexidade e generalização.
- O uso de Dropout nas camadas totalmente conectadas foi essencial para conter overfitting.
- O aumento progressivo do número de filtros teve impacto positivo na capacidade da rede aprender características visuais.

---

## 🖼️ Visualizações

O projeto também exibe:

- Filtros da primeira camada convolucional
- Exemplos de predições corretas e incorretas

---

## 📁 Estrutura do Projeto

```
cnn-cifar10-atividade4/
├── cnn_cifar10.py           # Código principal com definições da CNN e treinos
├── README.md                # Este arquivo
└── requirements.txt         # Dependências do projeto
```

---

## 👨‍🏫 Autoria

Atividade desenvolvida para a disciplina de **Redes Neurais Artificiais** - PPGCC UNESP.