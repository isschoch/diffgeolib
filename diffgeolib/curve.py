import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from typing import Callable
intervalTy = tuple[float, float]

class curve:
    def __init__(self, curveFct: Callable, interval: intervalTy = (0.0, 2.0 * jnp.pi)):
        self.curveFct = curveFct
        self.interval = interval
    
    def __call__(self, t: float):
        return self.curveFct(t)

    def prime(self, val, n=1):
        result = self.curveFct
        for i in range(n):
            result = jax.jacfwd(result)
        return result(val)
    
    def frenetFrame(self, t: float):
        def __gram_schmidt(input):
            results = input
            dim = np.shape(input)[1]
            num_args = np.shape(input)[0]

            for k in range(num_args):
                for j in range(k):
                    results[k] -= np.inner(results[j], input[k]) / np.linalg.norm(results[j])**2 * results[j]

            for k in range(num_args):
                results[k] /= np.linalg.norm(results[k])
            
            return results

        grammed_vecs = __gram_schmidt(np.array([jnp.asarray(self.prime(t)), jnp.asarray(self.prime(t, n=2))]))
        e_0 = grammed_vecs[0, :]
        e_1 = grammed_vecs[1, :]
        e_2 = np.cross(e_0, e_1)
        return e_0, e_1, e_2
    
    def plotCurve(self, interval=None, num_t_vals=41):
        if interval == None:
            interval = self.interval
        t_vals = jnp.linspace(interval[0], interval[1], num_t_vals)
        curve_vals = np.zeros(shape=(num_t_vals, 3))

        for i in range(num_t_vals):
            curve_vals[i, :] = self.curveFct(t_vals[i])
        
        ax = plt.axes(projection='3d')
        ax.plot3D(curve_vals[:, 0], curve_vals[:, 1], curve_vals[:, 2])
        ax.set_aspect("equal")

        return ax.get_figure(), ax

    def plotFrenetFrame(self, interval=None, num_t_vals=41, t=None):
        fig, ax = self.plotCurve(interval, num_t_vals)
        if t == None:
            t = (interval[0] + interval[1]) / 2.0
        e_0, e_1, e_2 = self.frenetFrame(t)        

        ax.plot3D(*self.curveFct(t), marker="o", zorder=100, color='b')
        plot_0 = ax.quiver3D(*self.curveFct(t), *e_0, color='black')
        ax.quiver3D(*self.curveFct(t), *e_1, color='red')
        ax.quiver3D(*self.curveFct(t), *e_2, color='green')

        return ax.get_figure(), ax
    
    def slidablePlotFrenetFrame(self, interval=None, num_t_vals=101):
        @widgets.interact(t=(self.interval[0], self.interval[1], (self.interval[1] - self.interval[0])/num_t_vals))
        def plot_t_val_coords(t):
            fig, ax = self.plotFrenetFrame(num_t_vals=num_t_vals, t=t)
