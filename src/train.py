import torch
import time
import os
from tqdm import tqdm
import numpy as np
from src.dataset import core_data_loader
from src.model import DistillModel, StudentLSTMModule

train_loader, test_loader = core_data_loader(eta=0.01, batch_size=1)

print('train_loader: ', len(train_loader))
print('test_loader: ', len(test_loader))

device = "cuda" if torch.cuda.is_available() else "cpu"

radius_matrix = np.random.rand(10, 10)
radius_matrix[radius_matrix < 0.3] = 0
print('radius_matrix: ', radius_matrix.flatten())

student_model = StudentLSTMModule(input_size=10, hidden_size=128, output_size=10, num_layer=2)
model = DistillModel(radius_matrix, student_model, expand_method='sin')
optimizer = torch.optim.Adam(model.student_model.parameters(), lr=0.001)

epochs = 100
with tqdm(total=epochs * len(train_loader)) as pbar:
    model.train()
    running_loss = 0.0
    # with torch.enable_grad():
    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        loss = model(inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.update(1)
        pbar.set_description(f'loss: {running_loss:.4f}')