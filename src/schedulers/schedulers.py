import math
from typing import Optional, Dict, Any, Callable
import matplotlib.pyplot as plt
import numpy as np

class LRScheduler:
    """
    Clase base para los learning rate schedulers
    """
    def __init__(self, max_lr: float, min_lr: float, verbose: bool = False):
        """
        Inicializa el scheduler base
        
        Args:
            max_lr: Tasa de aprendizaje máxima
            min_lr: Tasa de aprendizaje mínima
            verbose: Si se deben imprimir mensajes de validación
        """
        self._validate_lr_params(max_lr, min_lr)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.verbose = verbose
        
    def _validate_lr_params(self, max_lr: float, min_lr: float) -> None:
        """
        Valida los parámetros básicos de la tasa de aprendizaje
        
        Args:
            max_lr: Tasa de aprendizaje máxima
            min_lr: Tasa de aprendizaje mínima
            
        Raises:
            ValueError: Si los parámetros no son válidos
        """
        if max_lr <= 0:
            raise ValueError(f"max_lr debe ser positivo, se recibió {max_lr}")
        
        if min_lr < 0:
            raise ValueError(f"min_lr debe ser no negativo, se recibió {min_lr}")
        
        if max_lr < min_lr:
            raise ValueError(f"max_lr ({max_lr}) debe ser mayor que min_lr ({min_lr})")
    
    def __call__(self, step: int) -> float:
        """
        Método para llamar al scheduler y obtener el learning rate
        
        Args:
            step: Paso actual de entrenamiento
            
        Returns:
            float: Tasa de aprendizaje para el paso actual
        """
        raise NotImplementedError("Las subclases deben implementar este método")
    
    def plot_schedule(self, total_steps: int, steps_per_epoch: int = None) -> None:
        """
        Genera un gráfico del schedule de learning rate
        
        Args:
            total_steps: Número total de pasos para visualizar
            steps_per_epoch: Pasos por época (para marcar las épocas en el eje x)
        """
        if self.__class__.__name__ == 'SuperTwistingLR':
            lrs = [self(i, j) for i, j in zip(range(total_steps), np.random.randn(total_steps))]
        else:
            lrs = [self(i) for i in range(total_steps)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(lrs)
        plt.xlabel('Pasos' if steps_per_epoch is None else 'Épocas')
        plt.ylabel('Learning Rate')
        plt.title(f'Schedule de Learning Rate: {self.__class__.__name__}')
        plt.grid(True)
        
        if steps_per_epoch is not None:
            # Convertir eje x a épocas
            epoch_ticks = np.arange(0, total_steps, steps_per_epoch)
            plt.xticks(epoch_ticks, np.arange(len(epoch_ticks)))
        
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig("lr_schedule.png")


class WarmupDecayLR(LRScheduler):
    """
    Scheduler de learning rate con warmup y decay
    """
    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
        decay_type: Optional[str] = None,
        final_div_factor: Optional[float] = None,
        verbose: bool = False
    ):
        """
        Inicializa el scheduler
        
        Args:
            max_lr: Tasa de aprendizaje máxima después del warmup
            min_lr: Tasa de aprendizaje mínima al final del entrenamiento
            warmup_steps: Número de pasos para el warmup lineal
            max_steps: Número total de pasos de entrenamiento
            decay_type: Tipo de decay schedule ('cosine', 'linear', 'exponential')
            final_div_factor: Si se proporciona, anula min_lr para que sea max_lr / final_div_factor
            verbose: Si se deben imprimir mensajes de validación
        """
        # Inicializar la clase base
        super().__init__(max_lr, min_lr, verbose)
        
        # Validar parámetros adicionales
        self._validate_steps_params(warmup_steps, max_steps)
        self._validate_decay_type(decay_type)
        
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.decay_type = decay_type
        
        # Manejar final_div_factor
        if final_div_factor is not None:
            if final_div_factor <= 1:
                raise ValueError(f"final_div_factor debe ser mayor que 1, se recibió {final_div_factor}")
            self.min_lr = self.max_lr / final_div_factor
            if self.verbose:
                print(f"Estableciendo min_lr a {self.min_lr} basado en final_div_factor de {final_div_factor}")
    
    def _validate_steps_params(self, warmup_steps: int, max_steps: int) -> None:
        """
        Valida los parámetros relacionados con los pasos
        
        Args:
            warmup_steps: Número de pasos para el warmup
            max_steps: Número total de pasos
            
        Raises:
            ValueError: Si los parámetros no son válidos
        """
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps debe ser no negativo, se recibió {warmup_steps}")
        
        if max_steps <= 0:
            raise ValueError(f"max_steps debe ser positivo, se recibió {max_steps}")
        
        if warmup_steps >= max_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) debe ser menor que max_steps ({max_steps})"
            )
    
    def _validate_decay_type(self, decay_type: str) -> None:
        """
        Valida el tipo de decay
        
        Args:
            decay_type: Tipo de decay
            
        Raises:
            ValueError: Si el tipo de decay no es válido
        """
        valid_types = ['cosine', 'linear', 'exponential']
        if decay_type not in valid_types:
            raise ValueError(
                f"decay_type debe ser uno de {valid_types}, se recibió {decay_type}"
            )
    
    def __call__(self, step: int) -> float:
        """
        Calcula la tasa de aprendizaje para un paso dado
        
        Args:
            step: Paso actual de entrenamiento
            
        Returns:
            float: Tasa de aprendizaje para el paso actual
        """
        # Convertir paso a float para seguridad en los cálculos
        step_f = float(step)
        
        # 1) Warmup lineal durante warmup_steps pasos
        if step < self.warmup_steps:
            return self.max_lr * (step_f + 1) / float(self.warmup_steps)
        
        # 2) Si paso > max_steps, devolver tasa de aprendizaje mínima
        if step >= self.max_steps:
            return self.min_lr
            
        # 3) Entre medias, usar el decay especificado hasta la tasa de aprendizaje mínima
        decay_ratio = (step_f - self.warmup_steps) / float(self.max_steps - self.warmup_steps)
        
        # Comprobación de seguridad (debería estar garantizada por las condiciones anteriores)
        decay_ratio = max(0.0, min(1.0, decay_ratio))
        
        # Calcular coeficiente según el tipo de decay
        if self.decay_type == 'cosine':
            # Decay coseno (transición suave de max_lr a min_lr)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        elif self.decay_type == 'linear':
            # Decay lineal de max_lr a min_lr
            coeff = 1.0 - decay_ratio
        elif self.decay_type == 'exponential':
            # Decay exponencial
            coeff = math.exp(-decay_ratio * 3)  # El factor 3 controla la velocidad de decay
        else:
            # Nunca debería llegar aquí debido a la validación, pero por si acaso
            coeff = 1.0
        
        return self.min_lr + coeff * (self.max_lr - self.min_lr)


class SuperTwistingLR(LRScheduler):
    """
    Implementación del scheduler Super Twisting (incompleta según el código proporcionado)
    """
    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        rho: float = 0.5,
        verbose: bool = False
    ):
        """
        Inicializa el scheduler Super Twisting
        
        Args:
            max_lr: Tasa de aprendizaje máxima
            min_lr: Tasa de aprendizaje mínima
            rho: Cota de la perturbación
            verbose: Si se deben imprimir mensajes de validación
        """
        super().__init__(max_lr, min_lr, verbose)
        self.rho = rho
        self.loss_integral = 0.0
        self.val_loss = None
        
    def update_val_loss(self, val_loss):
        self.val_loss = val_loss
        
    def __call__(self, step: int, loss: float) -> float:
        loss_c = loss.item()
        
        # Si no tenemos pérdida de validación aún, usar solo la pérdida de entrenamiento
        if self.val_loss is None:
            s = loss_c
        else:
            # La superficie de deslizamiento es la diferencia entre las pérdidas
            s = 10*loss_c + 1*self.val_loss
        
        # Acumular la integral del error
        self.loss_integral += s
        
        # Calcular el learning rate usando la ley de control Super Twisting
        v = 1.1 * self.rho * np.sign(self.loss_integral)
        lr = 1.5 * self.rho * np.abs(s) ** (0.5) * np.sign(s) + v
        
        # Limitar el learning rate al rango especificado
        lr = max(self.min_lr, min(lr, self.max_lr))
        
        return lr

# Factory para crear schedulers
def create_scheduler(config: Dict[str, Any]) -> LRScheduler:
    """
    Crea un scheduler basado en la configuración proporcionada
    
    Args:
        config: Diccionario con la configuración del scheduler
        
    Returns:
        LRScheduler: Una instancia del scheduler configurado
    """
    scheduler_type = config.get('type', 'warmup_decay')
    
    if scheduler_type == 'warmup_decay':
        return WarmupDecayLR(
            max_lr=config.get('max_lr', 0.1),
            min_lr=config.get('min_lr', 1e-5),
            warmup_steps=config.get('warmup_steps', 0),
            max_steps=config.get('max_steps', 10000),
            decay_type=config.get('decay_type', 'cosine'),
            final_div_factor=config.get('final_div_factor'),
            verbose=config.get('verbose', False)
        )
    elif scheduler_type == 'super_twisting':
        return SuperTwistingLR(
            max_lr=config.get('max_lr', 0.1),
            min_lr=config.get('min_lr', 1e-5),
            rho=config.get('rho', 0.5),
            verbose=config.get('verbose', False)
        )
    else:
        raise ValueError(f"Tipo de scheduler no soportado: {scheduler_type}")


# Función auxiliar para mantener compatibilidad con el código anterior
def get_lr_scheduler(
    max_lr: float, 
    min_lr: float, 
    warmup_steps: int, 
    max_steps: int, 
    decay_type: str = 'cosine',
    final_div_factor: Optional[float] = None,
    verbose: bool = False
) -> Callable[[int], float]:
    """
    Crea un scheduler de learning rate con warmup y decay (función compatible con la versión anterior)
    
    Args:
        max_lr: Tasa de aprendizaje máxima después del warmup
        min_lr: Tasa de aprendizaje mínima al final del entrenamiento
        warmup_steps: Número de pasos para el warmup lineal
        max_steps: Número total de pasos de entrenamiento
        decay_type: Tipo de decay schedule ('cosine', 'linear', 'exponential')
        final_div_factor: Si se proporciona, anula min_lr para que sea max_lr / final_div_factor
        verbose: Si se deben imprimir mensajes de validación
        
    Returns:
        Callable[[int], float]: Función que mapea el número de paso a la tasa de aprendizaje
    """
    scheduler = WarmupDecayLR(
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        decay_type=decay_type,
        final_div_factor=final_div_factor,
        verbose=verbose
    )
    
    return scheduler


if __name__ == '__main__':
    
    scheduler = create_scheduler({
        'type': 'warmup_decay',
        'max_lr': 0.5,
        'min_lr': 1e-6,
        'warmup_steps': 1000,
        'max_steps': 10000,
        'decay_type': 'cosine',
        'final_div_factor': None,
        'rho': 0.005,
        'verbose': True
    })
    
    if hasattr(scheduler, 'rho'):
        print("La clase tiene un atributo llamado 'rho'.")
    else:
        print("La clase NO tiene un atributo llamado 'rho'.")
    
    scheduler.plot_schedule(total_steps=10000, steps_per_epoch=1000)
    
    