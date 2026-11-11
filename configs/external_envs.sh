# Source before `lrbench suite`: one Python per environment (see `lrbench models --md`).
# Paths are examples; point them at the venvs you built on the cluster.
export LRBENCH_PY_SO3LR=${LRBENCH_PY_SO3LR:-$HOME/venvs/so3lr/bin/python}   # so3lr 0.2.0 + jax[cuda12]<0.11
export LRBENCH_PY_MACE=${LRBENCH_PY_MACE:-$HOME/venvs/mace/bin/python}      # mace-torch[polar]>=0.3.16
# MACE checkpoints download on first use; keep the cache on shared storage and pre-fetch on a login node:
#   XDG_CACHE_HOME=$SCRATCH/cache $LRBENCH_PY_MACE -c "from mace.calculators import mace_polar, mace_mp; \
#     mace_polar(model='polar-1-m', device='cpu'); mace_mp(model='mh-1', head='omol', device='cpu')"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCRATCH/cache}
unset NVIDIA_TF32_OVERRIDE
