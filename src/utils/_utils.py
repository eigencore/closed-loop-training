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
