#*******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022, 2023 T.J. Sego
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
    file(GLOB PY_EXAMPLES "py/*.py")

    file(COPY ${PY_EXAMPLES} DESTINATION ${TF_PYPKG_DIR}/examples)
    install(FILES ${PY_EXAMPLES} DESTINATION ${TF_SITEPACKAGES}/tissue_forge/examples)

    file(GLOB IPYNB_EXAMPLES "py/examples/*.ipynb")
    file(COPY ${IPYNB_EXAMPLES} DESTINATION ${TF_PYPKG_DIR}/examples/notebooks)
    install(FILES ${IPYNB_EXAMPLES} DESTINATION ${TF_SITEPACKAGES}/tissue_forge/examples/notebooks)
endif()
