
def predict_on_grid(model, grid_size=20):
    """Predice valores en una cuadrícula para visualización"""
    model.eval()
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    with torch.no_grad():
        for i in range(grid_size):
            for j in range(grid_size):
                inp = torch.FloatTensor([[X[i, j], Y[i, j]]])
                Z[i, j] = model(inp).item()

    return Z