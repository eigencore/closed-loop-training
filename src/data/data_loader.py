import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_cifar10(batch_size=32, num_workers=4, data_dir='.'):
    """
    Carga el conjunto de datos CIFAR-10 y crea cargadores de datos
    para entrenamiento, validación y prueba
    
    Args:
        batch_size: Tamaño del lote
        num_workers: Número de workers para cargar los datos
        data_dir: Directorio donde se almacenarán los datos
        
    Returns:
        train_loader: DataLoader para el conjunto de entrenamiento
        val_loader: DataLoader para el conjunto de validación
        test_loader: DataLoader para el conjunto de prueba
        train_size: Número de pasos (batches) por época
    """
    # Definir estadísticas para normalización
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Transformaciones para el conjunto de entrenamiento (con aumentación de datos)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Transformaciones para el conjunto de validación/prueba (sin aumentación)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Descargar y cargar los datos de entrenamiento
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Separar el conjunto de entrenamiento en entrenamiento y validación
    val_size = 5000
    train_size = len(train_dataset) - val_size
    
    # Usar un generador con semilla fija para reproducibilidad
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # Asegurarnos de que los conjuntos tienen las transformaciones correctas
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    
    # Descargar y cargar los datos de prueba
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    steps_per_epoch = len(train_loader)
    
    return train_loader, val_loader, test_loader, steps_per_epoch