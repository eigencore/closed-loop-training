import torch
import numpy as np
from functions.target_function import target_function

def generate_multimodal_dataset(n_samples=1000, noise_level=0.05):
    """Genera datos para la función multimodal"""
    print("Generando dataset con superficie multimodal...")
    
    # Generar puntos de entrada uniformemente distribuidos
    x = np.random.uniform(-2, 2, n_samples)
    y = np.random.uniform(-2, 2, n_samples)
    
    # Calcular valores objetivo
    z = np.array([target_function(x_i, y_i) for x_i, y_i in zip(x, y)])
    
    # Añadir ruido gaussiano
    z += np.random.normal(0, noise_level, n_samples)
    
    # Crear la matriz de características
    X = np.column_stack((x, y))
    
    print(f"Dataset generado: {X.shape} características, {z.shape} objetivos")
    return X, z

def prepare_data(X, y, test_size=0.2, batch_size=32):
    """Prepara los datos para entrenamiento"""
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    # Dividir datos
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Convertir a tensores
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    # Crear DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor