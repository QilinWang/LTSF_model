import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import importlib

def reload_modules(modules_to_reload):
  """
  Reloads a list of specified modules using importlib.reload().

  Args:
    modules_to_reload: A list or tuple of module objects to reload.
  """
  print("Reloading modules...")
  reloaded_modules = set() # Keep track to avoid duplicate messages if listed twice
  for module in modules_to_reload:
    if module is not None and hasattr(module, '__name__'): # Basic check if it's a module
        if module.__name__ not in reloaded_modules:
          try:
            importlib.reload(module)
            print(f"  Reloaded: {module.__name__}")
            reloaded_modules.add(module.__name__)
          except Exception as e:
            print(f"  Failed to reload {module.__name__}: {e}")
        else:
            # Handle case where module appears multiple times in input list
            pass # Already reloaded in this call
    else:
      print(f"  Skipping invalid item: {module}")
  print("Module reload complete.")




def plot_lorenz96_3d(trajectory, dim_to_plot=[0, 1, 2]):
    """
    Plots the Lorenz-96 system dynamics in 3D phase space.

    Args:
        trajectory (torch.Tensor): Trajectory of the Lorenz-96 system (shape: [steps, dim]).
        dim_to_plot (list): Dimensions to visualize in 3D (e.g., [0, 1, 2]).
    """
    if len(dim_to_plot) != 3:
        raise ValueError("dim_to_plot must contain exactly 3 dimensions for a 3D plot.")
    
    trajectory = trajectory.cpu().numpy()  # Convert to NumPy array for plotting
    x, y, z = trajectory[:, dim_to_plot[0]], trajectory[:, dim_to_plot[1]], trajectory[:, dim_to_plot[2]]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x, y, z, lw=0.8)
    ax.set_xlabel(f'Dimension {dim_to_plot[0]}')
    ax.set_ylabel(f'Dimension {dim_to_plot[1]}')
    ax.set_zlabel(f'Dimension {dim_to_plot[2]}')
    ax.set_title('3D Phase Space Trajectory')
    
    plt.show()