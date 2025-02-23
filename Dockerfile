FROM ghcr.io/methods-group/proximalgalerkin:v0.2.0-alpha AS pg


# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
# 24.04 adds `ubuntu` as uid 1000;
# remove it if it already exists before creating our user
RUN id -nu ${NB_UID} && userdel --force $(id -nu ${NB_UID}) || true; \
    useradd -m ${NB_USER} -u ${NB_UID}
ENV HOME=/home/${NB_USER}

# Copy DOLFINx env and Firedrake env
RUN chown -R --no-preserve-root ${NB_USER}:${NB_USER} /dolfinx-env
RUN chown -R --no-preserve-root ${NB_USER}:${NB_USER} /firedrake-env
RUN cp -r /root/LVPP ${HOME}/LVPP
RUN chown -R --no-preserve-root ${NB_USER}:${NB_USER} ${HOME}/LVPP

# Copy over julia-env
RUN cp -r /root/.juliaup ${HOME}/.juliaup
RUN ln -sf ${HOME}/.juliaup/bin/julialauncher ${HOME}/.juliaup/bin/julia
RUN chown -R --no-preserve-root ${NB_USER}:${NB_USER} ${HOME}/.juliaup
ENV PATH=${HOME}/.juliaup/bin:${PATH}

#
# Copy home directory for usage in binder
WORKDIR ${HOME}
COPY --chown=${NB_UID} . ${HOME}

USER ${NB_USER}
RUN python3 -m pip install -e .
ENTRYPOINT []