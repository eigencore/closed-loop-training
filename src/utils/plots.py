def visualize_target_function():
    """Visualiza la función objetivo en 3D"""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear una malla de puntos para visualización
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = target_function(X[i, j], Y[i, j])
    
    # Graficar superficie
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Añadir colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Superficie de la Función Objetivo - Con Múltiples Valles y Regiones Inestables', fontsize=14)
    
    plt.savefig('target_function_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # También mostrar un contour plot para visualizar mejor los valles
    plt.figure(figsize=(12, 10))
    cp = plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(cp)
    plt.title('Contorno de la Función Objetivo - Observe los Múltiples Valles y Regiones Abruptas', fontsize=14)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    
    # Añadir anotaciones para destacar regiones problemáticas
    plt.annotate('Región con gradientes\nmuy pronunciados', xy=(0.6, 0.8), xytext=(1.0, 1.5),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2), fontsize=12)
    
    plt.annotate('Mínimo local', xy=(-0.8, -0.7), xytext=(-1.5, -1.5),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=2), fontsize=12)
    
    plt.annotate('Pico estrecho', xy=(0.7, -0.8), xytext=(1.5, -1.5),
                arrowprops=dict(facecolor='green', shrink=0.05, width=2), fontsize=12)
    
    plt.savefig('target_function_contour.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualizar el gradiente
    dy, dx = np.gradient(Z)
    grad_magnitude = np.sqrt(dx**2 + dy**2)
    
    plt.figure(figsize=(12, 10))
    cp = plt.contourf(X, Y, grad_magnitude, 50, cmap='hot')
    plt.colorbar(cp)
    plt.title('Magnitud del Gradiente - Observe las Regiones con Gradientes Extremos', fontsize=14)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    
    # Añadir anotaciones para destacar regiones con gradientes extremos
    plt.annotate('Gradientes muy grandes\n(pueden causar explosión)', xy=(0.7, 0.8), xytext=(1.5, 1.5),
                arrowprops=dict(facecolor='yellow', shrink=0.05, width=2), fontsize=12)
    
    plt.annotate('Gradientes casi planos\n(progreso muy lento)', xy=(-0.5, -0.5), xytext=(-1.5, -1.0),
                arrowprops=dict(facecolor='cyan', shrink=0.05, width=2), fontsize=12)
    
    plt.savefig('target_function_gradient.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def compare_learning_rates(X, y, lr_values=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1], epochs=200, batch_size=32, use_clipping=False, scheduler_type='cosine', scheduler_params={'T_max': 200, 'eta_min': 0.01 * 0.1}):
    """Compara diferentes tasas de aprendizaje fijas"""
    # Preparar datos
    train_loader, test_loader, X_train, y_train, X_test, y_test = prepare_data(
        X, y, batch_size=batch_size)

    results = {}
    criterion = nn.MSELoss()

    for lr in lr_values:
        print(f"\n--- Entrenando con learning rate: {lr} ---")

        # Crear modelo con la misma inicialización para comparación justa
        torch.manual_seed(42)
        model = SimpleNet()

        # Entrenar modelo, con o sin gradient clipping
        clip_value = 1.0 if use_clipping else None
        result = train_model(
            model,
            train_loader,
            (X_test, y_test),
            criterion,
            learning_rate=lr,
            epochs=epochs,
            clip_value=clip_value,
            scheduler_type=scheduler_type,
            scheduler_params=scheduler_params
        )

        # Evaluar en conjunto de prueba si el entrenamiento no divergió
        model.eval()
        try:
            with torch.no_grad():
                test_outputs = model(X_test)
                if torch.isnan(test_outputs).any() or torch.isinf(test_outputs).any():
                    print(
                        f"Modelo con LR={lr} produjo predicciones inválidas (NaN/Inf)")
                    test_loss = float('nan')
                else:
                    test_loss = criterion(test_outputs, y_test).item()
                    print(f"MSE final en test: {test_loss:.4f}")
                result['test_mse'] = test_loss

                # Generar predicciones en una cuadrícula para visualizar superficie aprendida
                try:
                    pred_grid = predict_on_grid(model)
                    result['final_predictions'] = pred_grid
                except:
                    print(
                        f"No se pudo generar cuadrícula de predicciones para LR={lr}")
                    result['final_predictions'] = None
        except:
            print(
                f"Error al evaluar modelo con LR={lr}, probablemente divergió")
            result['test_mse'] = float('nan')
            result['final_predictions'] = None

        # Almacenar modelo y resultados
        result['model'] = model
        results[lr] = result

    return results



def visualize_predictions(results, lr_values):
    """Visualiza las predicciones finales de cada modelo con diferente learning rate"""
    # Crear la cuadrícula de puntos reales
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z_true = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_true[i, j] = target_function(X[i, j], Y[i, j])

    # Filtrar learning rates con predicciones válidas
    valid_lrs = [lr for lr in lr_values if results[lr]
                 ['final_predictions'] is not None]

    if len(valid_lrs) == 0:
        print("Ningún modelo produjo predicciones válidas para visualizar.")
        return

    # Graficar superficie real y predicciones
    fig = plt.figure(figsize=(18, 12))

    # Superficie verdadera
    ax1 = fig.add_subplot(231, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_true, cmap='viridis', alpha=0.8)
    ax1.set_title('Superficie Real', fontsize=12)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Predicciones para cada learning rate
    for i, lr in enumerate(valid_lrs[:5]):  # Mostrar hasta 5 learning rates
        if i >= 5:
            break

        ax = fig.add_subplot(232 + i, projection='3d')

        # Obtener predicciones del grid
        Z_pred = results[lr]['final_predictions']

        # Redimensionar si es necesario
        if Z_pred.shape[0] != X.shape[0]:
            from scipy.interpolate import griddata
            grid_x, grid_y = np.mgrid[-2:2:Z_pred.shape[0]
                                      * 1j, -2:2:Z_pred.shape[1]*1j]
            points = np.column_stack((grid_x.flatten(), grid_y.flatten()))
            values = Z_pred.flatten()
            Z_pred_interp = griddata(points, values, (X, Y), method='cubic')
            Z_pred = Z_pred_interp

        surf = ax.plot_surface(X, Y, Z_pred, cmap='plasma', alpha=0.8)
        ax.set_title(f'Predicción con LR={lr}', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig('prediction_comparison_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Graficar curvas de entrenamiento y normas de gradiente
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    for lr in lr_values:
        train_losses = np.array(results[lr]['train_losses'])
        # Limitar el rango para mejor visualización
        train_losses = np.clip(train_losses, 0, 10)
        plt.plot(train_losses, label=f'LR={lr}')
    plt.title('Pérdida de Entrenamiento (limitada a [0,10])', fontsize=12)
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    for lr in lr_values:
        val_losses = np.array(results[lr]['val_losses'])
        # Limitar el rango para mejor visualización
        val_losses = np.clip(val_losses, 0, 10)
        plt.plot(val_losses, label=f'LR={lr}')
    plt.title('Pérdida de Validación (limitada a [0,10])', fontsize=12)
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    for lr in lr_values:
        grad_norms = np.array(results[lr]['gradient_norms'])
        # Limitar el rango para mejor visualización
        grad_norms = np.clip(grad_norms, 0, 20)
        plt.plot(grad_norms, label=f'LR={lr}')
    plt.title('Norma del Gradiente (limitada a [0,20])', fontsize=12)
    plt.xlabel('Épocas')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.grid(True)

    # Gráfico de barras del error final
    plt.subplot(2, 2, 4)
    test_errors = [results[lr]['test_mse'] for lr in lr_values]
    # Reemplazar NaN por un valor grande para visualización
    test_errors = [10 if np.isnan(err) else min(err, 10)
                   for err in test_errors]

    plt.bar(range(len(lr_values)), test_errors, color='skyblue')
    plt.xticks(range(len(lr_values)), [
               f'LR={lr}' for lr in lr_values], rotation=45)
    plt.ylabel('MSE en Test (limitado a 10)')
    plt.title('Error Final por Learning Rate', fontsize=12)

    for i, v in enumerate(test_errors):
        if v == 10:
            plt.text(i, v - 1, "Divergió", ha='center',
                     color='red', fontweight='bold')
        else:
            plt.text(i, v + 0.5, f'{v:.4f}', ha='center')

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_with_without_clipping(X, y, lr_values=[1e-3, 1e-2, 5e-2], epochs=200, batch_size=32):
    """Compara entrenamiento con y sin gradient clipping para diferentes learning rates"""
    print("\n=== Entrenamiento SIN Gradient Clipping ===")
    results_no_clip = compare_learning_rates(
        X, y, lr_values, epochs, batch_size, use_clipping=False)

    print("\n=== Entrenamiento CON Gradient Clipping ===")
    results_with_clip = compare_learning_rates(
        X, y, lr_values, epochs, batch_size, use_clipping=True)

    # Visualizar comparación de pérdidas
    plt.figure(figsize=(15, 10))

    for i, lr in enumerate(lr_values):
        plt.subplot(len(lr_values), 2, 2*i+1)

        # Pérdida de entrenamiento sin clipping
        train_losses_no_clip = np.array(results_no_clip[lr]['train_losses'])
        train_losses_no_clip = np.clip(train_losses_no_clip, 0, 10)
        plt.plot(train_losses_no_clip, label='Sin Clipping', color='red')

        # Pérdida de entrenamiento con clipping
        train_losses_with_clip = np.array(
            results_with_clip[lr]['train_losses'])
        train_losses_with_clip = np.clip(train_losses_with_clip, 0, 10)
        plt.plot(train_losses_with_clip, label='Con Clipping', color='blue')

        plt.title(f'Pérdida de Entrenamiento (LR={lr})', fontsize=10)
        plt.xlabel('Épocas')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)

        plt.subplot(len(lr_values), 2, 2*i+2)

        # Norma del gradiente sin clipping
        grad_norms_no_clip = np.array(results_no_clip[lr]['gradient_norms'])
        grad_norms_no_clip = np.clip(grad_norms_no_clip, 0, 20)
        plt.plot(grad_norms_no_clip, label='Sin Clipping', color='red')

        # Norma del gradiente con clipping
        grad_norms_with_clip = np.array(
            results_with_clip[lr]['gradient_norms'])
        grad_norms_with_clip = np.clip(grad_norms_with_clip, 0, 20)
        plt.plot(grad_norms_with_clip, label='Con Clipping', color='blue')

        plt.title(f'Norma del Gradiente (LR={lr})', fontsize=10)
        plt.xlabel('Épocas')
        plt.ylabel('L2 Norm')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('clipping_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results_no_clip, results_with_clip