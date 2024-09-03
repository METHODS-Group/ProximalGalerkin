FROM ghcr.io/fenics/dolfinx/dolfinx:nightly

WORKDIR /galahad
RUN python3 -m pip install meson

RUN git clone https://github.com/ralna/GALAHAD.git
ENV MESON_DIR build-dir-galahad
WORKDIR /galahad/GALAHAD
RUN meson setup ${MESON_DIR} -Dpythoniface=true -Dpython.install_env=auto && \
    meson compile -C ${MESON_DIR} && \
    meson install -C ${MESON_DIR} && \
    meson test -C ${MESON_DIR} --suite=Python
ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}
WORKDIR /root