FROM ghcr.io/fenics/dolfinx/dev-env:v0.9.0-openmpi AS galahad
ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

WORKDIR /galahad
RUN python3 -m pip install meson

ARG GALAHAD_VERSION=v5.2.0
RUN git clone --single-branch --branch=${GALAHAD_VERSION} https://github.com/ralna/GALAHAD.git
ENV MESON_DIR=build-dir-galahad
WORKDIR /galahad/GALAHAD
RUN meson setup ${MESON_DIR} -Dpythoniface=true -Dpython.install_env=auto && \
    meson compile -C ${MESON_DIR} && \
    meson install -C ${MESON_DIR} && \
    meson test -C ${MESON_DIR} --suite=Python

ARG IPOPT_VERSION=3.14.11

# Pull MUMPS 
WORKDIR /ipopt
RUN git clone https://github.com/coin-or-tools/ThirdParty-Mumps --single-branch --branch stable/3.0


# Build MUMPS for correct architecture
WORKDIR /ipopt/ThirdParty-Mumps
RUN ./get.Mumps;
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; \
    then ./configure --build=aarch64-unknown-linux-gnu --with-lapack-lflags="-llapack -lblas" --prefix="/usr/local"; \
    elif [ "$TARGETPLATFORM" = "linux/amd64" ]; \
    then ./configure --build=amd64-unknown-linux-gnu --with-lapack-lflags="-llapack -lblas" --prefix="/usr/local"  ; \
    else echo "Unknown TARGETPLATFORM: ${TARGETPLATFORM}, it should be either 'linux/amd64' or 'linux/arm64'"; \
    fi
RUN make && \
    make install

FROM galahad AS optimization
WORKDIR /ipopt

# Pull IPOPT 
RUN git clone https://github.com/coin-or/Ipopt --single-branch --branch releases/${IPOPT_VERSION}

# Build Ipopt for correct architecture
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; \
    then ./Ipopt/configure --build=aarch64-unknown-linux-gnu --with-lapack-lflags="-llapack -lblas" --prefix="/usr/local" --enable-debug --enable-shared --with-mumps-cflags="-I/usr/local/include/coin-or/mumps" --with-mumps-lflags="-L/usr/local/lib -lcoinmumps" ; \
    elif [ "$TARGETPLATFORM" = "linux/amd64" ]; \
    then ./Ipopt/configure --build=amd64-unknown-linux-gnu --with-lapack-lflags="-llapack -lblas" --prefix="/usr/local" --enable-debug --enable-shared --with-mumps-cflags="-I/usr/local/include/coin-or/mumps" --with-mumps-lflags="-L/usr/local/lib -lcoinmumps" ; \
    else echo "Unknown TARGETPLATFORM: ${TARGETPLATFORM}, it should be either 'linux/amd64' or 'linux/arm64'"; \
    fi
RUN make install


# Build pyipopt
ARG CYIPOPT_VERSION=1.5.0
RUN python3 -m pip install --no-cache git+https://github.com/mechmotum/cyipopt.git@v${CYIPOPT_VERSION}

ARG PYVISTA_VERSION=0.44.2
ARG DOLFINX_VERSION=v0.9.0.post1
ARG BASIX_VERSION=v0.9.0
ARG UFL_VERSION=2024.2.0
ARG FFCX_VERSION=v0.9.0

FROM optimization AS dolfinx-onbuild
LABEL description="DOLFINx in 32-bit real and complex modes (onbuild)"

ONBUILD WORKDIR /src

ONBUILD RUN git clone --branch=${DOLFINX_VERSION} --single-branch https://github.com/fenics/dolfinx.git
ONBUILD RUN git clone --branch=${FFCX_VERSION} --single-branch https://github.com/fenics/ffcx.git
ONBUILD RUN git clone --branch=${BASIX_VERSION} --single-branch https://github.com/fenics/basix.git
ONBUILD RUN git clone --branch=${UFL_VERSION} --single-branch https://github.com/fenics/ufl.git


ONBUILD RUN cp ./dolfinx/docker/dolfinx-real-mode /usr/local/bin/dolfinx-real-mode
ONBUILD RUN cp ./dolfinx/docker/dolfinx-complex-mode /usr/local/bin/dolfinx-complex-mode
ONBUILD RUN chmod +x /usr/local/bin/dolfinx-*-mode


# These files are empty by default, i.e. they do nothing.
# The user can set them at build time if they wish.
ONBUILD RUN mkdir -p /root/.config/dolfinx /root/.config/ffcx
ONBUILD RUN cp dolfinx/docker/dolfinx_jit_options.json /root/.config/dolfinx/dolfinx_jit_options.json
ONBUILD RUN cp dolfinx/docker/ffcx_options.json /root/.config/ffcx/ffcx_options.json

# CMake build type for DOLFINx C++ build. See CMake documentation.
ONBUILD ARG DOLFINX_CMAKE_BUILD_TYPE="Release"

# Using pip install `.[test]` with --no-dependencies and --no-build-isolation
# does not install necessary packages, hence install build and optional
# dependencies manually here.
ONBUILD RUN pip install --no-cache-dir -r dolfinx/python/build-requirements.txt && \
    pip install --no-cache-dir pyamg pytest scipy matplotlib numba # test + optional set

# The dolfinx-onbuild container expects to have folders basix/ ufl/
# ffcx/ and dolfinx/ mounted/shared at /src.
ONBUILD RUN cd basix && cmake -G Ninja -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -B build-dir -S ./cpp && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    pip install ./python && \
    cd ../ufl && pip install --no-cache-dir . && \
    cd ../ffcx && pip install --no-cache-dir . && \
    cd ../ && pip install --no-cache-dir ipython

# --no-dependencies necessary as --target does not contain any dependencies e.g.
# mpi4py - leading to unwanted rebuild.
ONBUILD RUN cd dolfinx && \
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

# Real by default.
ONBUILD ENV PKG_CONFIG_PATH=/usr/local/dolfinx-real/lib/pkgconfig:$PKG_CONFIG_PATH \
    CMAKE_PREFIX_PATH=/usr/local/dolfinx-real/lib/cmake:$CMAKE_PREFIX_PATH \
    PETSC_ARCH=linux-gnu-real64-32 \
    PYTHONPATH=/usr/local/dolfinx-real/lib/python3.12/dist-packages:$PYTHONPATH \
    LD_LIBRARY_PATH=/usr/local/dolfinx-real/lib:$LD_LIBRARY_PATH

ONBUILD WORKDIR /root

########################################

FROM dolfinx-onbuild AS intermediate


FROM optimization AS dolfinx-user-env
LABEL description="DOLFINx in 32-bit real and complex modes"

########################################


COPY --from=intermediate /usr/local /usr/local
COPY --from=intermediate /root/.config /root/.config
COPY --from=intermediate /dolfinx-env /dolfinx-env
COPY --from=intermediate /src /src


ENV PKG_CONFIG_PATH=/usr/local/dolfinx-real/lib/pkgconfig:$PKG_CONFIG_PATH \
    CMAKE_PREFIX_PATH=/usr/local/dolfinx-real/lib/cmake:$CMAKE_PREFIX_PATH \
    PETSC_ARCH=linux-gnu-real64-32 \
    PYTHONPATH=/usr/local/dolfinx-real/lib/python3.12/dist-packages:$PYTHONPATH \
    LD_LIBRARY_PATH=/usr/local/dolfinx-real/lib:$LD_LIBRARY_PATH


WORKDIR /root

RUN pip install --no-cache-dir jupyter jupyterlab

# pyvista dependencies from apt
RUN apt-get -qq update && \
    apt-get -y install libgl1-mesa-dev xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install pyvista from PyPI. pyvisa depends on (py)vtk), but vtk wheels are not
# available on pypi for linux/arm64, so we use a custom build wheel.
# matplotlib improves plotting quality with better color maps and
# properly rendering colorbars.
# trame is the preferred backend for pyvista.
RUN dpkgArch="$(dpkg --print-architecture)"; \
    pip install matplotlib; \
    case "$dpkgArch" in amd64) \
    pip install --no-cache-dir pyvista[jupyter]==${PYVISTA_VERSION} ;; \
    esac; \
    case "$dpkgArch" in arm64) \
    pip install --no-cache-dir https://github.com/scientificcomputing/vtk-aarch64/releases/download/v9.4.1-py3.12/vtk-9.4.1.dev0-cp312-cp312-linux_aarch64.whl && \
    pip install --no-cache-dir pyvista[jupyter]==${PYVISTA_VERSION} ;; \
    esac; \
    pip cache purge

# Jupyter Notebook kernel specification for complex build DOLFINx
RUN mkdir -p /usr/local/share/jupyter/kernels/python3-complex/
RUN cp /src/dolfinx/docker/complex-kernel.json /usr/local/share/jupyter/kernels/python3-complex/kernel.json



EXPOSE 8888/tcp
ENV SHELL /bin/bash

WORKDIR /root

FROM dolfinx-user-env AS firedrake-petsc-env
RUN apt-get update && apt-get install -y curl sudo

ENV TMP_PYTHON_PATH=${PYTHONPATH}
ENV TMP_PETSC_DIR=${PETSC_DIR}
ENV TMP_PETSC_ARCH=${PETSC_ARCH}

ENV PYTHONPATH=""
ENV HDF5_DIR="/usr/local"
ENV HDF5_MPI="ON"
ARG FIREDRAKE_BRANCH="2025.4.1"
# Setup firedrake virtual env
ENV FIREDRAKE_PYTHON=/firedrake-env
RUN python3 -m venv ${FIREDRAKE_PYTHON}
WORKDIR ${FIREDRAKE_PYTHON}

RUN git clone --branch ${FIREDRAKE_BRANCH} --single-branch https://github.com/firedrakeproject/firedrake.git

# Build recipe from firedrake website
ENV PATH=${FIREDRAKE_PYTHON}/bin:$PATH

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y liblapack-dev
# Install firedrake-dependent PETSc
RUN git clone --branch $(python3 ./firedrake/scripts/firedrake-configure --show-petsc-version) https://gitlab.com/petsc/petsc.git
WORKDIR ${FIREDRAKE_PYTHON}/petsc
ENV PETSC_DIR=${FIREDRAKE_PYTHON}/petsc
ENV PETSC_ARCH=arch-firedrake-default
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y autoconf libtool flex bison libhwloc-dev libfftw3-dev libfftw3-mpi-dev
ENV HDF5_DIR="/usr/local"
ENV HDF5_MPI="ON"
RUN ./configure  --with-c2html=0 --with-debugging=0 --with-fortran-bindings=0 --with-shared-libraries=1 \
    --with-strict-petscerrorcode --COPTFLAGS=-O3 -march=native -mtune=native --CXXOPTFLAGS=-O3 -march=native -mtune=native --FOPTFLAGS=-O3 -march=native \
    -mtune=native --with-bison --with-fftw --with-hdf5-lib=${HDF5_DIR}/lib/libhdf5.a --with-hdf5-include="${HDF5_DIR}/include" --with-hwloc --download-metis --download-mumps \
    --download-netcdf --download-pnetcdf --download-ptscotch --download-scalapack --download-suitesparse --download-superlu_dist --with-zlib --download-hypre
RUN make PETSC_ARCH=${PETSC_ARCH} -j${BUILD_NP} all

FROM firedrake-petsc-env AS firedrake-env

# INSTALL Firedrake
WORKDIR ${FIREDRAKE_PYTHON}/firedrake
RUN python3 -m pip install --no-cache-dir mpi4py petsc4py matplotlib -v
RUN python3 -m pip install --no-binary h5py . -v
# Install ngs-petsc and vtk on amd, only vtk on arm
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; \
    then python3 -m pip install --no-cache-dir https://github.com/scientificcomputing/vtk-aarch64/releases/download/v9.4.1-py3.12/vtk-9.4.1.dev0-cp312-cp312-linux_aarch64.whl; \
    elif [ "$TARGETPLATFORM" = "linux/amd64" ]; \
    then python3 -m pip install ngspetsc netgen-mesher vtk ; \
    else echo "Unknown TARGETPLATFORM: ${TARGETPLATFORM}, it should be either 'linux/amd64' or 'linux/arm64'"; \
    fi

# Revert to DOLFINx environment
ENV PYTHONPATH=${TMP_PYTHON_PATH}
ENV PETSC_DIR=${TMP_PETSC_DIR}
ENV PETSC_ARCH=${TMP_PETSC_ARCH}

FROM firedrake-env AS mfem-user-env
WORKDIR /root/LVPP

ARG HYPRE_VERSION=2.32.0
ARG METIS_VERSION=4.0.3
ARG MFEM_BRANCH=v4.8

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && \
    apt-get -y install wget git cmake build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /root/LVPP

# Build hypre
RUN wget https://github.com/hypre-space/hypre/archive/refs/tags/v${HYPRE_VERSION}.tar.gz
RUN tar -xvf v${HYPRE_VERSION}.tar.gz

RUN cd hypre-${HYPRE_VERSION}/src/ && \
    ./configure --disable-fortran && \
    make -j 2 && \
    cd ../.. && \
    ln -s hypre-${HYPRE_VERSION} hypre

# Build metis
RUN wget https://github.com/mfem/tpls/raw/gh-pages/metis-${METIS_VERSION}.tar.gz
RUN ls
RUN tar -xvf metis-${METIS_VERSION}.tar.gz
RUN cd metis-${METIS_VERSION} && \
    make OPTFLAGS=-Wno-error=implicit-function-declaration && \
    cd .. && \
    ln -s metis-${METIS_VERSION} metis-4.0

# Build mfem
RUN git clone --branch=${MFEM_BRANCH} --single-branch https://github.com/mfem/mfem.git
RUN cd mfem && \
    make parallel -j 2

FROM mfem-user-env AS julia-user-env

RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://install.julialang.org | sh  -s -- --yes
ENV PATH=/root/.juliaup/bin:$PATH
RUN julia -e 'using Pkg; Pkg.add(["ClassicalOrthogonalPolynomials", "MultivariateOrthogonalPolynomials", "Plots", "LaTeXStrings", "IterativeSolvers", "LinearMaps", "MatrixFactorizations", "Gridap", "LineSearches"])'



# ADD various files to the docker container for easy switching between environments
WORKDIR /root/LVPP/mfem/examples
ADD ./examples/09_eikonal/ex40.cpp ex40_taylor_hood.cpp
ADD ./examples/09_eikonal/convert_mesh.cpp convert_mesh.cpp
ADD docker/firedrake-mode /usr/local/bin/firedrake-mode
RUN chmod +x /usr/local/bin/firedrake-mode
ADD docker/dolfinx-mode /usr/local/bin/dolfinx-mode
RUN chmod +x /usr/local/bin/dolfinx-mode

# Add all examples in repository
WORKDIR /root/LVPP
ADD . .

# Build DOLFINx dependencies
# netgen-mesher does not have a linux/arm build
ENV PATH=/dolfinx-env/bin:$PATH
WORKDIR /root/LVPP
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; \
    then python3 -m pip install -e .[obstacle,signorini,eikonal,monge_ampere]; \
    elif [ "$TARGETPLATFORM" = "linux/amd64" ]; \
    then python3 -m pip install -e .[examples]  ; \
    else echo "Unknown TARGETPLATFORM: ${TARGETPLATFORM}, it should be either 'linux/amd64' or 'linux/arm64'"; \
    fi
ENV OMP_NUM_THREADS=1

