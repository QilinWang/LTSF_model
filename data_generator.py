import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, Optional, Any, List, Callable


class DynamicalSystem(ABC):
    """Abstract base class for dynamical systems."""
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Return default parameters for this dynamical system."""
        return cls._default_params.copy()
    
    @abstractmethod
    def generate(self, steps: int, dt: float, initial_cond: Any, **kwargs) -> torch.Tensor:
        """Generate a trajectory for this dynamical system."""
        pass
    
    @classmethod
    def create(cls, steps: int = 50000, dt: float = 0.01, 
              initial_cond: Any = None, params: Dict[str, Any] = None) -> torch.Tensor:
        """Factory method to create an instance and generate a trajectory."""
        # Get default parameters
        default_params = cls.get_default_params()
        
        # Override with user parameters if provided
        if params:
            default_params.update(params)
        
        # Use provided initial conditions or defaults
        if initial_cond is not None:
            default_params['initial_cond'] = initial_cond
        
        # Create instance and generate
        system = cls(device=default_params.get('device', 'cuda'))
        return system.generate(
            steps=steps,
            dt=dt,
            **default_params
        )

# === Standalone RK4 Step Function ===
def rk4_step(
    current_state: torch.Tensor,
    deriv_func: Callable[..., torch.Tensor],
    dt: float,
    params: Dict[str, Any]
) -> torch.Tensor:
    """
    Performs a single RK4 integration step.

    Args:
        current_state (torch.Tensor): The current state vector.
        deriv_func (Callable): A function that computes the derivatives
                                (e.g., _lorenz_derivs, _rossler_derivs).
                                Its signature should be func(state, **params).
        dt (float): The time step size.
        params (Dict[str, Any]): Dictionary containing parameters needed by deriv_func.

    Returns:
        torch.Tensor: The state vector after one RK4 step.
    """
    k1 = deriv_func(current_state,                 **params)
    k2 = deriv_func(current_state + (k1 * dt / 2.0), **params)
    k3 = deriv_func(current_state + (k2 * dt / 2.0), **params)
    k4 = deriv_func(current_state + (k3 * dt),       **params)

    new_state = current_state + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0)
    return new_state

class LorenzSystem(DynamicalSystem):
    """
    Lorenz chaotic system generator.
    Uses abstracted RK4 step function.
    """
    
    _default_params = {
        'sigma': 10.0,
        'rho': 28.0,
        'beta': 8.0 / 3.0,
        'initial_cond': (1.0, 0.98, 1.1),
        'device': 'cuda',
        'method': 'rk4'
    }
    
    def __init__(self, device: str = 'cuda', method: str = 'rk4'):
        super().__init__()
        self.device = device
        if method not in ['euler', 'rk4']:
            raise ValueError("Method must be 'euler' or 'rk4'")
        self.method = method
        print(f"LorenzSystem initialized with method: {self.method} on device: {self.device}")

    @staticmethod
    def _lorenz_derivs(state: torch.Tensor, sigma: float, rho: float, beta: float) -> torch.Tensor:
        """Calculates the derivatives dx/dt, dy/dt, dz/dt for the Lorenz system."""
        x, y, z = state[0], state[1], state[2]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return torch.stack([dx, dy, dz])

    def generate(self, steps: int, dt: float, initial_cond: Tuple[float, float, float], 
                sigma: float, rho: float, beta: float, **kwargs) -> torch.Tensor:
        """Generate a Lorenz system trajectory."""
        current_state = torch.tensor(initial_cond, dtype=torch.float32, device=self.device)

        
        # Initialize trajectory storage
        trajectory = torch.empty((steps + 1, 3), dtype=torch.float32, device=self.device)
        trajectory[0] = current_state
        
        # Store parameters needed by the derivative function 
        deriv_params = {'sigma': sigma, 'rho': rho, 'beta': beta}

        # Generate trajectory
        for step in range(1, steps + 1):
            # Compute derivatives
            if self.method == 'euler':
                derivs = self._lorenz_derivs(current_state, sigma, rho, beta)
                current_state += derivs * dt
            elif self.method == 'rk4':
                current_state = rk4_step(
                    current_state,
                    self._lorenz_derivs, # Pass the derivative function
                    dt,
                    deriv_params        # Pass the necessary parameters
                )
                # # --- RK4 Method ---
                # k1 = self._lorenz_derivs(current_state,                 sigma, rho, beta)
                # k2 = self._lorenz_derivs(current_state + (k1 * dt / 2.0), sigma, rho, beta)
                # k3 = self._lorenz_derivs(current_state + (k2 * dt / 2.0), sigma, rho, beta)
                # k4 = self._lorenz_derivs(current_state + (k3 * dt),       sigma, rho, beta)

                # current_state += (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0)
                # ------------------
            trajectory[step] = current_state
        # Combine dimensions 
        return trajectory





class RosslerSystem(DynamicalSystem):
    """
    Rössler chaotic system (continuous ODEs).

    Can use either Forward Euler ('euler') or 4th Order Runge-Kutta ('rk4') method.
    """
    
    _default_params = {
        'a': 0.2,
        'b': 0.2,
        'c': 5.7,
        'initial_cond': (1.0, 1.0, 1.0),
        'device': 'cuda',
        'method': 'rk4'
    }
    
    def __init__(self, device: str = 'cuda', method: str = 'rk4'):
        """
        Initializes the Rössler system generator.

        Args:
            device (str): The torch device ('cuda' or 'cpu').
            method (str): The integration method ('euler' or 'rk4'). Defaults to 'rk4'.
        """
        super().__init__() 
        self.device = device
        if method not in ['euler', 'rk4']:
            raise ValueError("Method must be 'euler' or 'rk4'")
        self.method = method
        print(f"RosslerSystem initialized with method: {self.method} on device: {self.device}")
    
    @staticmethod
    def _rossler_derivs(state: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
        """Calculates the derivatives dx/dt, dy/dt, dz/dt for the Rössler system."""
        x, y, z = state[0], state[1], state[2]
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        return torch.stack([dx, dy, dz])

    
    def generate(self, steps: int, dt: float, initial_cond: Tuple[float, float, float],
                 a: float, b: float, c: float, **kwargs) -> torch.Tensor:
        """
        Generate a Rössler system trajectory using the selected method.

        Args:
            steps (int): Number of integration steps.
            dt (float): Time step size.
            initial_cond (Tuple[float, float, float]): Initial state (x, y, z).
            a (float): Rössler parameter a.
            b (float): Rössler parameter b.
            c (float): Rössler parameter c.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            torch.Tensor: Tensor containing the trajectory of shape (steps + 1, 3).
        """
        current_state = torch.tensor(initial_cond, dtype=torch.float32, device=self.device) 
        # Initialize trajectory storage
        trajectory = torch.empty((steps + 1, 3), dtype=torch.float32, device=self.device)
        trajectory[0] = current_state 

        deriv_params = {'a': a, 'b': b, 'c': c}
        # Generate trajectory
        for step in range(1, steps + 1):
            if self.method == 'euler':
                # --- Euler Method ---
                derivs = self._rossler_derivs(current_state, a, b, c)
                current_state += derivs * dt

            elif self.method == 'rk4':
                # --- RK4 Method ---
                current_state = rk4_step(
                    current_state,
                    self._rossler_derivs, # Pass the derivative function
                    dt,
                    deriv_params         # Pass the necessary parameters
                )
                # k1 = self._rossler_derivs(current_state,                 a, b, c)
                # k2 = self._rossler_derivs(current_state + (k1 * dt / 2.0), a, b, c)
                # k3 = self._rossler_derivs(current_state + (k2 * dt / 2.0), a, b, c)
                # k4 = self._rossler_derivs(current_state + (k3 * dt),       a, b, c)

                # current_state += (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0)
            trajectory[step] = current_state
        # Combine dimensions
        return trajectory


class HyperchaosticRosslerSystem(DynamicalSystem):
    """Hyperchaotic Rössler system with four dimensions."""
    
    _default_params = {
        'a': 0.25, 'b': 3.0, 'c': 0.5, 'd': 0.05,
        'initial_cond': (1.0, 1.0, 4.0, 1.0), 'device': 'cuda', 'method': 'rk4'
    }
    
    def __init__(self, device: str = 'cuda', method: str = 'rk4'):
        super().__init__()
        self.device = device
        if method not in ['euler', 'rk4']: raise ValueError("Method must be 'euler' or 'rk4'")
        self.method = method
        print(f"HyperchaoticRosslerSystem initialized with method: {self.method} on device: {self.device}")

    @staticmethod
    def _hyper_rossler_derivs(state: torch.Tensor, a: float, b: float, c: float, d: float) -> torch.Tensor:
        """Calculates the derivatives for the Hyperchaotic Rössler system."""
        x, y, z, w = state[0], state[1], state[2], state[3]
        dx = -y - z
        dy = x + a * y + w
        dz = b + x * z
        dw = -c * z + d * w # Corrected based on common formulations (check if your intended equations differ)
        # Original had dw = c*w + d*x, which might be a different variant. Using the more standard one here.
        # If dw = c*w + d*x was intended, change the line above accordingly.
        return torch.stack([dx, dy, dz, dw])
    
    def generate(self, steps: int, dt: float, initial_cond: Tuple[float, float, float, float],
                 a: float, b: float, c: float, d: float, **kwargs) -> torch.Tensor:
        """Generate a Hyperchaotic Rössler system trajectory."""
        current_state = torch.tensor(initial_cond, dtype=torch.float32, device=self.device)
        trajectory = torch.empty((steps + 1, 4), dtype=torch.float32, device=self.device)
        trajectory[0] = current_state
        deriv_params = {'a': a, 'b': b, 'c': c, 'd': d}
        for step in range(1, steps + 1):
            if self.method == 'euler':
                derivs = self._hyper_rossler_derivs(current_state, **deriv_params)
                current_state += derivs * dt
            elif self.method == 'rk4':
                current_state = rk4_step(current_state, self._hyper_rossler_derivs, dt, deriv_params)

            # Check for numerical instability (optional but good practice)
            if torch.isnan(current_state).any() or torch.isinf(current_state).any():
                print(f"Numerical instability at step {step}: state={current_state}")
                # Handle as needed: break, raise error, or try smaller dt
                raise ValueError("Numerical instability detected.")

            trajectory[step] = current_state
        return trajectory





class DuffingSystem(DynamicalSystem):
    """Duffing oscillator chaotic system (continuous ODEs)."""
    _default_params = {
        'alpha': 1.0, 'beta': -1.0, 'delta': 0.2, 'gamma': 0.3, 'omega': 1.0,
        'initial_cond': (0.1, 0.1), 'device': 'cuda', 'method': 'rk4'
    } 
    
    def __init__(self, device: str = 'cuda', method: str = 'rk4'):
        super().__init__()
        self.device = device
        if method not in ['euler', 'rk4']: raise ValueError("Method must be 'euler' or 'rk4'")
        self.method = method
        print(f"DuffingSystem initialized with method: {self.method} on device: {self.device}")

    
    @staticmethod
    def _duffing_derivs(state: torch.Tensor, t: float, alpha: float, beta: float, delta: float, gamma: float, omega: float) -> torch.Tensor:
        """Calculates the derivatives dx/dt, dv/dt for the Duffing system."""
        # Note: Duffing equation is often time-dependent due to forcing term
        x, v = state[0], state[1]
        dx = v
        dv = gamma * torch.cos(omega * t) - delta * v - alpha * x - beta * x**3
        return torch.stack([dx, dv])
    
    def generate(self, steps: int, dt: float, initial_cond: Tuple[float, float],
                 alpha: float, beta: float, delta: float, gamma: float, omega: float,
                 **kwargs) -> torch.Tensor:
        """Generate a Duffing oscillator trajectory."""
        current_state = torch.tensor(initial_cond, dtype=torch.float32, device=self.device)
        trajectory = torch.empty((steps + 1, 2), dtype=torch.float32, device=self.device)
        trajectory[0] = current_state
        # Time needs to be tracked for the forcing term
        current_time = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        deriv_params = {'alpha': alpha, 'beta': beta, 'delta': delta, 'gamma': gamma, 'omega': omega}

        for step in range(1, steps + 1):
            # Pass current time and state to derivative function
            # Need a wrapper or modification if using the generic rk4_step
            # Let's implement RK4 explicitly here due to time dependency

            if self.method == 'euler':
                # Pass current_time along with state and params
                derivs = self._duffing_derivs(current_state, current_time, **deriv_params)
                current_state += derivs * dt
            elif self.method == 'rk4':
                # RK4 for non-autonomous systems (depends explicitly on time t)
                t_n = current_time
                y_n = current_state

                # Need to pass time 't' to the derivative function
                k1 = self._duffing_derivs(y_n,             t_n,          **deriv_params)
                k2 = self._duffing_derivs(y_n + k1*dt/2.0, t_n + dt/2.0, **deriv_params)
                k3 = self._duffing_derivs(y_n + k2*dt/2.0, t_n + dt/2.0, **deriv_params)
                k4 = self._duffing_derivs(y_n + k3*dt,     t_n + dt,     **deriv_params)

                current_state += (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0)

            # Update time
            current_time += dt

            # Check for numerical instability
            if torch.isnan(current_state).any() or torch.isinf(current_state).any():
                print(f"Numerical instability at step {step}: state={current_state}, t={current_time}")
                raise ValueError("Numerical instability detected.")

            trajectory[step] = current_state
        return trajectory


class Lorenz96System(DynamicalSystem):
    """Lorenz-96 chaotic system."""
    _default_params = {
        'dim': 6, 'forcing': 8.0, 'initial_cond': None,
        'reduce_dim': None, 'method': 'rk4', 'device': 'cuda'
    } 
    
    def __init__(self, device: str = 'cuda', method: str = 'rk4'):
        super().__init__()
        self.device = device
        if method not in ['euler', 'rk4']: raise ValueError("Method must be 'euler' or 'rk4'")
        self.method = method
        print(f"Lorenz96System initialized with method: {self.method} on device: {self.device}")

    @staticmethod
    def _lorenz96_derivs(state: torch.Tensor, forcing: float) -> torch.Tensor:
        """Calculates the derivatives for the Lorenz-96 system."""
        x_roll_p1 = torch.roll(state, shifts=-1, dims=0)  # x_{i+1}
        x_roll_m1 = torch.roll(state, shifts=1, dims=0)   # x_{i-1}
        x_roll_m2 = torch.roll(state, shifts=2, dims=0)   # x_{i-2}
        dxdt = (x_roll_p1 - x_roll_m2) * x_roll_m1 - state + forcing
        return dxdt
    
    def generate(self, steps: int, dt: float, dim: int, forcing: float,
                 initial_cond: Optional[torch.Tensor] = None,
                 reduce_dim: Optional[int] = None,
                 **kwargs) -> torch.Tensor:
        """Generate a Lorenz-96 system trajectory."""
        assert dim > 3, "dim must be greater than 3 for standard Lorenz96"
        if initial_cond is None:
            # Standard initialization: F everywhere, perturb one variable
            state = torch.full((dim,), forcing, dtype=torch.float32, device=self.device)
            state[0] += 0.01 # Small perturbation
        elif isinstance(initial_cond, (list, tuple)):
             state = torch.tensor(initial_cond, dtype=torch.float32, device=self.device)
        elif isinstance(initial_cond, torch.Tensor):
             state = initial_cond.to(dtype=torch.float32, device=self.device)
        else:
             raise TypeError("initial_cond must be Tensor, list, tuple, or None")

        if state.shape[0] != dim:
             raise ValueError(f"initial_cond dimension ({state.shape[0]}) must match dim ({dim})")

        # Use steps+1 to include initial condition
        trajectory = torch.empty((steps + 1, dim), dtype=torch.float32, device=self.device)
        trajectory[0] = state

        deriv_params = {'forcing': forcing} # Parameters needed by deriv func

        for step in range(1, steps + 1):
            if self.method == 'euler':
                derivs = self._lorenz96_derivs(state, **deriv_params)
                state += derivs * dt
            elif self.method == 'rk4':
                state = rk4_step(state, self._lorenz96_derivs, dt, deriv_params)

            # Check for instability
            if torch.isnan(state).any() or torch.isinf(state).any():
                print(f"Numerical instability at step {step}: state max abs={state.abs().max()}")
                raise ValueError("Numerical instability detected.")

            trajectory[step] = state

        # Optionally reduce dimension (simple linear projection example)
        if reduce_dim is not None and reduce_dim < dim:
            # Simple projection matrix (e.g., take first `reduce_dim` components)
            # More sophisticated reduction like PCA could be used offline
            trajectory = trajectory[:, :reduce_dim]
            print(f"Reduced trajectory dimension to {reduce_dim}")

        return trajectory

# === Henon System (Discrete Map - No changes needed for RK4 abstraction) ===

class HenonSystem(DynamicalSystem):
    """
    Henon map chaotic system (discrete map).
    Note: Integration methods like 'euler' or 'rk4' do not apply to discrete maps.
    """

    
    _default_params = {
        'a': 1.4,
        'b': 0.3,
        'initial_cond': (0.0, 0.0),
        'device': 'cuda'
    }
    
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        print(f"HenonSystem initialized on device: {self.device}")
    
    def generate(self, steps: int, dt: float, initial_cond: Tuple[float, float], 
                a: float, b: float, **kwargs) -> torch.Tensor:
        """
        Generate a Henon map trajectory.

        Args:
            steps (int): Number of iterations.
            dt (Optional[float]): Time step size (ignored for discrete maps).
            initial_cond (Tuple[float, float]): Initial state (x, y).
            a (float): Henon parameter a.
            b (float): Henon parameter b.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            torch.Tensor: Tensor containing the trajectory of shape (steps + 1, 2).
        """
        # For discrete systems like Henon, dt is not used
        current_state = torch.tensor(initial_cond, dtype=torch.float32, device=self.device)
 
        trajectory = torch.empty((steps + 1, 2), dtype=torch.float32, device=self.device)
        trajectory[0] = current_state
        # Generate trajectory by iterating the map
        x, y = current_state[0], current_state[1]
        for step in range(1, steps + 1):
            x_new = 1.0 - a * x**2 + y
            y_new = b * x 
            x, y = x_new, y_new 
            current_state = torch.stack([x, y]) 
            trajectory[step] = current_state

        return trajectory
    
# === Logistic System (Discrete Map - No changes needed for RK4 abstraction) ===

class LogisticSystem(DynamicalSystem):
    """Logistic map chaotic system."""
    _default_params = { 'r': 3.9, 'initial_cond': 0.5, 'device': 'cuda' }
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        print(f"LogisticSystem initialized on device: {self.device}")
    
    def generate(self, steps: int, dt: Optional[float], initial_cond: float,
                 r: float, **kwargs) -> torch.Tensor:
        x = torch.tensor(initial_cond, dtype=torch.float32, device=self.device)
        # Need steps+1 points to include initial condition
        trajectory_x = torch.empty(steps + 1, dtype=torch.float32, device=self.device)
        trajectory_x[0] = x
        for step in range(1, steps + 1):
            x = r * x * (1.0 - x) # Ensure float arithmetic
            trajectory_x[step] = x
        # Return shape (steps+1, 1) for consistency
        return trajectory_x.unsqueeze(-1)

class TrajectoryFactory:
    """Factory for creating various dynamical system trajectories."""
    
    # Registry of available systems
    _systems = {
        'lorenz': LorenzSystem,
        'henon': HenonSystem,
        'rossler': RosslerSystem,
        'hyper_rossler': HyperchaosticRosslerSystem,
        'logistic': LogisticSystem,
        'duffing': DuffingSystem,
        'lorenz96': Lorenz96System
    }
    
    @classmethod
    def register_system(cls, name: str, system_class: DynamicalSystem) -> None:
        """Register a new dynamical system."""
        cls._systems[name] = system_class
    
    @classmethod
    def get_available_systems(cls) -> List[str]:
        """Get a list of available dynamical systems."""
        return list(cls._systems.keys())
    
    @classmethod
    def create_trajectory(cls, system: str, **kwargs) -> torch.Tensor:
        """
        Create a trajectory for the specified dynamical system.
        
        Args:
            system: Name of the dynamical system
            **kwargs: System-specific parameters
            
        Returns:
            torch.Tensor: Generated trajectory
        """
        if system not in cls._systems:
            available = ", ".join(cls._systems.keys())
            raise ValueError(f"Unknown system '{system}'. Available systems: {available}")
        
        # Extract common parameters
        steps = kwargs.pop('steps', 50399)
        dt = kwargs.pop('dt', 0.01)
        initial_cond = kwargs.pop('initial_cond', None)
        params = kwargs
        
        # Create the trajectory
        return cls._systems[system].create(steps=steps, dt=dt, 
            initial_cond=initial_cond, params=params)


# Compatibility function to match the original generate_trajectory API
def generate_trajectory(system, initial_cond=None, params=None, steps=50399, dt=0.01):
    """
    Generates trajectories for specified dynamical systems.
    
    Args:
        system (str): The system to generate ('lorenz', 'henon', 'rossler', 'logistic', 'duffing', etc.).
        initial_cond (tuple or float, optional): Initial conditions to override defaults.
        params (dict, optional): System-specific parameters to override defaults.
        steps (int, optional): Number of steps to simulate. Default is 49999.
        dt (float, optional): Time step size for continuous systems. Default is 0.01.
        
    Returns:
        torch.Tensor: The generated trajectory.
    """
    return TrajectoryFactory.create_trajectory(
        system=system,
        steps=steps,
        dt=dt,
        initial_cond=initial_cond,
        **(params or {})
    )


# Example usage:
if __name__ == "__main__":
    # Get a Lorenz trajectory
    lorenz_traj = generate_trajectory('lorenz', steps=10000)
    print(f"Lorenz trajectory shape: {lorenz_traj.shape}")
    
    # Get a Henon trajectory with custom parameters
    henon_traj = generate_trajectory('henon', 
                                    initial_cond=(0.1, 0.1), 
                                    params={'a': 1.2, 'b': 0.3}, 
                                    steps=5000)
    print(f"Henon trajectory shape: {henon_traj.shape}")
    
    # Using the factory directly
    lorenz96_traj = TrajectoryFactory.create_trajectory(
        'lorenz96', 
        steps=8000, 
        dim=10, 
        forcing=8.0
    )
    print(f"Lorenz-96 trajectory shape: {lorenz96_traj.shape}")
    
    # List available systems
    print(f"Available systems: {TrajectoryFactory.get_available_systems()}")