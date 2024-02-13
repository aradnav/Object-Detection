import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define matrices
x = torch.rand(1000, 1000).to(device)
y = torch.rand(1000, 1000).to(device)

# Perform matrix multiplication
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

start_time.record()
result = torch.mm(x, y)
end_time.record()

# Wait for computation to finish
torch.cuda.synchronize()

# Calculate elapsed time
elapsed_time = start_time.elapsed_time(end_time)
print("Elapsed time for matrix multiplication on CUDA: {} milliseconds".format(elapsed_time))
