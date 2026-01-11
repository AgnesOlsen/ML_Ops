import torch
from src.ml_ops_ex.model import MyAwesomeModel


def test_training_step():
    """Test that a single training step actually updates model weights."""
    model = MyAwesomeModel()

    # Capture weights before a training step
    # We check the very first layer (conv1) to see if it moves
    initial_weights = model.conv1.weight.clone()

    # Create dummy data (Batch size 2, 1 channel, 28x28)
    dummy_input = torch.randn(2, 1, 28, 28)
    dummy_target = torch.randint(0, 10, (2,))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Perform one forward and backward pass
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = loss_fn(output, dummy_target)
    loss.backward()
    optimizer.step()

    # Capture weights after the step
    updated_weights = model.conv1.weight.clone()

    # Assert that weights are no longer identical
    assert not torch.equal(initial_weights, updated_weights), "Model weights did not update after training step!"
    assert not torch.isnan(loss), "Loss is NaN - check your learning rate or data normalization!"


def test_model_output_range():
    """Test that model outputs the correct number of classes (10)."""
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10), f"Expected shape (1, 10), but got {y.shape}"
