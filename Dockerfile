FROM ghcr.io/fenics/dolfinx/dev-env:current

WORKDIR /tmp

ARG DOLFINX_VERSION=dokken/blocked_linear_solver
ARG BASIX_VERSION=main
ARG UFL_VERSION=main
ARG FFCX_VERSION=main



RUN git clone --branch=${DOLFINX_VERSION} --single-branch https://github.com/fenics/dolfinx.git
RUN git clone --branch=${FFCX_VERSION} --single-branch https://github.com/fenics/ffcx.git
RUN git clone --branch=${BASIX_VERSION} --single-branch https://github.com/fenics/basix.git
RUN git clone --branch=${UFL_VERSION} --single-branch https://github.com/fenics/ufl.git


RUN cd basix && cmake -G Ninja -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -B build-dir -S ./cpp && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    pip install ./python && \
    cd ../ufl && pip install --no-cache-dir . -U && \
    cd ../ffcx && pip install --no-cache-dir . -U && \
    cd ../ && pip install --no-cache-dir ipython

RUN pip install --no-cache-dir -r dolfinx/python/build-requirements.txt && \
    pip install --no-cache-dir pyamg pytest scipy matplotlib numba # test + optional set

RUN cd dolfinx && \
    mkdir -p build-real && \
    cd build-real && \
    PETSC_ARCH=linux-gnu-real64-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-real -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} ../cpp && \
    ninja install && \
    cd ../python && \
    PETSC_ARCH=linux-gnu-real64-32 pip -v install \
    --config-settings=cmake.build-type="${DOLFINX_CMAKE_BUILD_TYPE}" --config-settings=install.strip=false --no-build-isolation --check-build-dependencies \
    --target /usr/local/dolfinx-real/lib/python3.12/dist-packages --no-dependencies --no-cache-dir '.' && \
    git clean -fdx && \
    cd ../ && \
    mkdir -p build-complex && \
    cd build-complex && \
    PETSC_ARCH=linux-gnu-complex128-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} ../cpp && \
    ninja install && \
    . /usr/local/dolfinx-complex/lib/dolfinx/dolfinx.conf && \
    cd ../python && \
    PETSC_ARCH=linux-gnu-complex128-32 pip -v install \
    --config-settings=cmake.build-type="${DOLFINX_CMAKE_BUILD_TYPE}" --config-settings=install.strip=false --no-build-isolation --check-build-dependencies \
    --target /usr/local/dolfinx-complex/lib/python3.12/dist-packages --no-dependencies --no-cache-dir '.'




ENV PKG_CONFIG_PATH=/usr/local/dolfinx-real/lib/pkgconfig:$PKG_CONFIG_PATH \
    CMAKE_PREFIX_PATH=/usr/local/dolfinx-real/lib/cmake:$CMAKE_PREFIX_PATH \
    PETSC_ARCH=linux-gnu-real64-32 \
    PYTHONPATH=/usr/local/dolfinx-real/lib/python3.12/dist-packages:$PYTHONPATH \
    LD_LIBRARY_PATH=/usr/local/dolfinx-real/lib:$LD_LIBRARY_PATH