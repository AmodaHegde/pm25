import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Create a neural network
model = nn.Sequential(
    nn.Linear(3, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 2. Train the PINN
for epoch in range(3000):

    # Generate random x values
    x = torch.rand(100, 1, requires_grad=True)

    # Predict y for each x
    y = model(x)

    # Calculate dy/dx using automatic differentiation
    dy_dx = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

    # Physics loss: dy/dx + y should equal zero
    physics_loss = torch.mean((dy_dx + y) ** 2)

    # Initial condition: y(0) = 1
    y0 = model(torch.zeros(1, 1))
    initial_loss = (y0 - 1) ** 2

    # Combine both losses
    loss = physics_loss + initial_loss

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 3. Compare PINN with the exact solution
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)

with torch.no_grad():
    y_pinn = model(x_test)

y_exact = torch.exp(-x_test)

plt.plot(x_test.numpy(), y_exact.numpy(), label="Exact")
plt.plot(x_test.numpy(), y_pinn.numpy(), "--", label="PINN")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()