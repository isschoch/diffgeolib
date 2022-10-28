import diffgeolib as dgl
import jax.numpy as jnp

def fct(t):
    return jnp.array([t, t**2, t**3])

myCurve = dgl.curve(fct)
fig, ax = myCurve.plotCurve()
fig.show()

e0, e1, e2 = myCurve.frenetFrame(1.0)
