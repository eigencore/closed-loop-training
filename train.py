import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import math
from typing import Optional, Callable, Dict, List, Any, Tuple, Union
import os
import json

from src.data.data_loader import load_cifar10
from src.models.model import SimpleCNN, SmallResNet
from src.schedulers.schedulers import create_scheduler

def train_with_scheduler(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    scheduler: Union[Callable[[int, Optional[torch.Tensor]], float], str],
    optimizer_type: str = 'sgd',
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    max_lr: float = 0.1,
    min_lr: float = 1e-5,
    num_epochs: int = 100,
    device: torch.device = None,
    save_dir: str = './results',
    experiment_name: str = 'scheduler_experiment',
    save_checkpoints: bool = True
) -> Dict[str, List[float]]:
    """
    Entrena un modelo utilizando un scheduler de learning rate
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader para los datos de entrenamiento
        val_loader: DataLoader para los datos de validación
        scheduler: Scheduler personalizado o "cosine" para usar CosineAnnealingLR de PyTorch
        optimizer_type: Tipo de optimizador ('sgd' o 'adam')
        momentum: Momentum para SGD
        weight_decay: Weight decay para el optimizador
        max_lr: Learning rate máximo
        min_lr: Learning rate mínimo (usado solo para cosine de PyTorch)
        num_epochs: Número de épocas para entrenar
        device: Dispositivo donde ejecutar el entrenamiento (CPU o GPU)
        save_dir: Directorio para guardar resultados
        experiment_name: Nombre del experimento
        save_checkpoints: Si se deben guardar checkpoints del modelo
    
    Returns:
        history: Diccionario con métricas de entrenamiento
    """
    # Configurar dispositivo
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mover el modelo al dispositivo
    model = model.to(device)
    
    # Definir la función de pérdida
    criterion = nn.CrossEntropyLoss()
    
    # Crear optimizador base
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=max_lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizador no soportado: {optimizer_type}")
    
    # Configurar scheduler PyTorch si se especifica "cosine"
    use_pytorch_scheduler = isinstance(scheduler, str) and scheduler.lower() == 'cosine'
    if use_pytorch_scheduler:
        # Usar CosineAnnealingLR de PyTorch
        pytorch_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    
    # Crear directorio para guardar resultados
    os.makedirs(save_dir, exist_ok=True)
    experiment_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Guardar configuración del experimento
    config = {
        'optimizer_type': optimizer_type,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'max_lr': max_lr,
        'min_lr': min_lr,
        'num_epochs': num_epochs,
        'device': str(device),
        'model_type': model.__class__.__name__,
        'scheduler': 'CosineAnnealingLR (PyTorch)' if use_pytorch_scheduler else 'Custom Scheduler'
    }
    
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Variables para seguimiento
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    total_steps = 0
    steps_per_epoch = len(train_loader)
    max_steps = num_epochs * steps_per_epoch
    
    # Iniciar temporizador
    start_time = time.time()
    
    print(f"Iniciando entrenamiento en {device}")
    print(f"Total de pasos planificados: {max_steps} (épocas={num_epochs}, pasos por época={steps_per_epoch})")
    print(f"Usando scheduler: {'CosineAnnealingLR (PyTorch)' if use_pytorch_scheduler else 'Custom Scheduler'}")
    
    for epoch in range(num_epochs):
        # ===== FASE DE ENTRENAMIENTO =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Si estamos usando CosineAnnealingLR de PyTorch, obtener el learning rate actual
        if use_pytorch_scheduler:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Barra de progreso para la fase de entrenamiento
        train_loop = tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, (inputs, targets) in enumerate(train_loop):
            # Mover datos al dispositivo
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Actualizar learning rate para este paso si estamos usando un scheduler personalizado
            if not use_pytorch_scheduler:
                # Verificar si el scheduler necesita el loss como entrada
                if hasattr(scheduler, 'rho'):  # Caso para SuperTwistingLR
                    current_lr = scheduler(total_steps, loss)
                else:  # Caso para otros schedulers personalizados
                    current_lr = scheduler(total_steps)
                
                # Actualizar learning rate en el optimizador
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Actualizar estadísticas
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Actualizar barra de progreso
            if not use_pytorch_scheduler:
                current_lr = optimizer.param_groups[0]['lr']  # Obtener el LR actual
                
            train_loop.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'acc': 100. * train_correct / train_total,
                'lr': current_lr
            })
            
            # Incrementar contador de pasos
            total_steps += 1
        
        # Calcular métricas de entrenamiento para la época
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # ===== FASE DE VALIDACIÓN =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Actualizar la pérdida de validación en el scheduler si es SuperTwistingLR
        if not use_pytorch_scheduler and hasattr(scheduler, 'update_val_loss'):
            scheduler.update_val_loss(val_loss)
        
        # Barra de progreso para la fase de validación
        val_loop = tqdm(val_loader, desc=f"Época {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loop):
                # Mover datos al dispositivo
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Actualizar estadísticas
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Actualizar barra de progreso
                val_loop.set_postfix({
                    'loss': val_loss / (batch_idx + 1),
                    'acc': 100. * val_correct / val_total
                })
        
        # Calcular métricas de validación para la época
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Paso del scheduler PyTorch al final de cada época
        if use_pytorch_scheduler:
            pytorch_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']  # Obtener el LR actualizado
        
        # Guardar métricas en la historia
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Imprimir resumen de la época
        print(f"Época {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
              f"LR: {current_lr:.6f}")
        
        # Guardar el mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            
            if save_checkpoints:
                checkpoint_path = os.path.join(experiment_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"Mejor modelo guardado con precisión de validación: {best_val_acc:.2f}%")
    
    # Calcular tiempo total de entrenamiento
    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print(f"Entrenamiento completado en {hours}h {minutes}m {seconds}s")
    print(f"Mejor precisión de validación: {best_val_acc:.2f}%")
    
    # Guardar historial
    history_path = os.path.join(experiment_dir, 'history.json')
    with open(history_path, 'w') as f:
        # Convertir valores a lista para serialización JSON
        serializable_history = {k: [float(i) for i in v] for k, v in history.items()}
        json.dump(serializable_history, f, indent=4)
    
    # Guardar modelo final
    if save_checkpoints:
        final_path = os.path.join(experiment_dir, 'final_model.pth')
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss
        }, final_path)
    
    # Restaurar el mejor modelo en el modelo actual
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history

def plot_training_curves(history: Dict[str, List[float]], save_path: str = None):
    """
    Grafica curvas de entrenamiento a partir del historial
    
    Args:
        history: Diccionario con métricas de entrenamiento
        save_path: Ruta para guardar la figura
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Graficar pérdidas
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.set_title('Pérdida durante el entrenamiento')
    ax1.legend()
    ax1.grid(True)
    
    # Graficar precisiones
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión (%)')
    ax2.set_title('Precisión durante el entrenamiento')
    ax2.legend()
    ax2.grid(True)
    
    # Graficar learning rate
    ax3.plot(history['lr'])
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate durante el entrenamiento')
    ax3.grid(True)
    # Escala logarítmica para el LR
    ax3.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figura guardada en: {save_path}")
    
    plt.show()

def main():
    """Función principal para ejecutar el entrenamiento"""
    import argparse
    
    # Configurar parser para argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenamiento CIFAR-10 con scheduler')
    parser.add_argument('--model', type=str, default='simple_cnn', choices=['simple_cnn', 'small_resnet'],
                        help='Modelo a utilizar (default: simple_cnn)')
    parser.add_argument('--batch-size', type=int, default=128, help='Tamaño de batch (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas (default: 100)')
    parser.add_argument('--max-lr', type=float, default=0.1, help='Learning rate máximo (default: 0.1)')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='Learning rate mínimo (default: 1e-5)')
    parser.add_argument('--scheduler-type', type=str, default='cosine', 
                        choices=['cosine', 'super_twisting'],
                        help='Tipo de scheduler (default: cosine)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='Optimizador a utilizar (default: sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum para SGD (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay para el optimizador (default: 5e-4)')
    parser.add_argument('--rho', type=float, default=0.005,
                        help='Parámetro rho para Super Twisting (default: 0.005)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Nombre del experimento (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Si no se proporciona un nombre para el experimento, generar uno
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_{args.scheduler_type}_maxlr{args.max_lr}_epochs{args.epochs}"
    
    # Cargar datos
    print("Cargando datos CIFAR-10...")
    train_loader, val_loader, test_loader, steps_per_epoch = load_cifar10(batch_size=args.batch_size)
    
    # Crear modelo
    print(f"Creando modelo: {args.model}")
    if args.model == 'simple_cnn':
        model = SimpleCNN()
    else:  # small_resnet
        model = SmallResNet()
    
    # Configurar scheduler
    print(f"Configurando scheduler: {args.scheduler_type}")
    
    if args.scheduler_type == 'super_twisting':
        # Usar scheduler personalizado de Super Twisting
        scheduler = create_scheduler({
            'type': 'super_twisting',
            'max_lr': args.max_lr,
            'min_lr': args.min_lr,
            'rho': args.rho,
            'verbose': True
        })
    else:
        # Usar CosineAnnealingLR de PyTorch
        scheduler = 'cosine'  # Indicador para usar el scheduler de PyTorch
    
    # Entrenar modelo
    print(f"Iniciando entrenamiento: {args.epochs} épocas")
    history = train_with_scheduler(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        optimizer_type=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        num_epochs=args.epochs,
        experiment_name=args.experiment_name
    )
    
    # Graficar resultados
    print("Generando gráficas...")
    save_path = os.path.join('./results', args.experiment_name, 'training_curves.png')
    plot_training_curves(history, save_path)
    
    print("¡Entrenamiento completado!")

if __name__ == '__main__':
    main()