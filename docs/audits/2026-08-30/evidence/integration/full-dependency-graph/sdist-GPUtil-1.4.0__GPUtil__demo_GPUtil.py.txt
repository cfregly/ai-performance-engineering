import GPUtil as GPU
import sys
# Get all device ids and their processing and memory utiliazion
# (deviceIds, gpuUtil, memUtil) = GPU.getGPUs()

# Print os and python version information
print('OS: ' + sys.platform)
print(sys.version)

# Print package name and version number
print(GPU.__name__ + ' ' + GPU.__version__)

# Show the utilization of all GPUs in a nice table
GPU.showUtilization()

# Show all stats of all GPUs in a nice table
GPU.showUtilization(all=True)

# Get all available GPU(s), ordered by ID in ascending order
print('All available ordered by id: '),
print(GPU.getAvailable(order='first', limit=999))

# Get 1 available GPU, ordered by ID in descending order
print('Last available: '),
print(GPU.getAvailable(order='last', limit=1))

# Get 1 random available GPU
print('Random available: '),
print(GPU.getAvailable(order='random'))

# Get 1 available GPU, ordered by GPU load ascending
print('First available weighted by GPU load ascending: '),
print(GPU.getAvailable(order='load', limit=1))

# Get all available GPU with max load of 10%, ordered by memory ascending
print('All available weighted by memory load ascending: '),
print(GPU.getAvailable(order='memory', limit=999, maxLoad=0.1))

# Get the first available GPU
firstGPU = GPU.getFirstAvailable()
print('First available GPU id:'),
print(firstGPU)

# Get the first available GPU, where memory usage is less than 90% and processing is less than 80%
firstGPU = GPU.getFirstAvailable(maxMemory=0.9, maxLoad=0.8)
print('First available GPU id (memory < 90%, load < 80%):'),
print(firstGPU)

# Get the first available GPU, where processing is less than 1%
firstGPU = GPU.getFirstAvailable(attempts=5, interval=5, maxLoad=0.01, verbose=True)
print('First available GPU id (load < 1%):'),
print(firstGPU)
# NOTE: If all your GPUs currently have a load larger than 1%, this step will
# fail. It's not a bug! It is intended to do so, if it does not find an available GPU.

# Get the first available GPU, where memory usage is less than 1%
firstGPU = GPU.getFirstAvailable(attempts=5, interval=5, maxMemory=0.01, verbose=True)
print('First available GPU id (memory < 1%):'),
print(firstGPU)
# NOTE: If all your GPUs currently have a memory consumption larger than 1%,
# this step will fail. It's not a bug! It is intended to do so, if it does not
# find an available GPU.
