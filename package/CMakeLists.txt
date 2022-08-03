#*******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022 T.J. Sego
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
#*******************************************************************************

if(TF_WITH_PYTHON)

    # Determine install locations w.r.t. python module root
    file(REAL_PATH "${TF_SITEPACKAGES}" TF_SITEPACKAGES_REAL)
    file(RELATIVE_PATH TFPY_RPATH_BIN 
        ${TF_SITEPACKAGES_REAL}/tissue_forge 
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
    )
    set(TFPY_RPATH_BIN '${TFPY_RPATH_BIN}')
    file(RELATIVE_PATH TFPY_RPATH_LIB 
        ${TF_SITEPACKAGES_REAL}/tissue_forge 
        ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    )
    set(TFPY_RPATH_LIB '${TFPY_RPATH_LIB}')

    # These populate the python configuration script; overwite them accordingly per platform and package
    set(TFPY_RPATH_BIN_SITE None)
    set(TFPY_RPATH_LIB_SITE None)

    # Select package based on settings

    if(TF_PACKAGE_LOCAL)
        add_subdirectory(local)
    endif()
    if(TF_PACKAGE_CONDA)
        add_subdirectory(conda)
    endif()

endif(TF_WITH_PYTHON)