def target_function(x, y):
    """
    Función objetivo que combina múltiples mínimos locales con regiones
    de gradientes muy grandes (que pueden desestabilizar el entrenamiento)
    """
    # Base: función de Ackley modificada - conocida por ser difícil de optimizar
    # debido a sus numerosos mínimos locales y un gradiente casi plano
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    ackley = term1 + term2 + a + np.exp(1)
    
    # Añadir regiones con gradientes muy pronunciados (parte inestable)
    steep_region = 10 * np.tanh(5 * (x**2 + y**2 - 1.5)) + 5 * np.sin(10 * x * y)
    
    # Añadir picos altos y estrechos en ciertos puntos
    peaks = 8 * np.exp(-5 * ((x-0.7)**2 + (y+0.8)**2)) - 8 * np.exp(-5 * ((x+0.7)**2 + (y-0.8)**2))
    
    # Añadir valles irregulares adicionales
    valleys = 2 * np.sin(7 * x) * np.cos(7 * y) / (1 + 0.5 * (x**2 + y**2))
    
    # Combinar todo - normalizar para mantener los valores en un rango razonable
    result = (ackley / 10) + (steep_region / 5) + peaks + valleys
    
    
    return result