#*******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022-2024 T.J. Sego
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

file(RELATIVE_PATH TF_SITEPACKAGES_REL 
    ${CMAKE_INSTALL_PREFIX}/etc 
    ${TF_SITEPACKAGES}
)

file(RELATIVE_PATH PY_ROOT_DIR_REL
    ${CMAKE_INSTALL_PREFIX}/etc 
    ${PY_ROOT_DIR}
)

if(WIN32)
    set(VARFILE_DIR win)
    set(VARFILE_SUFFIX bat)
    set(TFPY_RPATH_BIN_SITE ${TFPY_RPATH_BIN})
elseif(APPLE)
    set(VARFILE_DIR osx)
    set(VARFILE_SUFFIX sh)
else()
    set(VARFILE_DIR linux)
    set(VARFILE_SUFFIX sh)
endif()

configure_file(${VARFILE_DIR}/site_vars.${VARFILE_SUFFIX}.in ${CMAKE_CURRENT_BINARY_DIR}/site_vars.${VARFILE_SUFFIX} @ONLY)
install(
    FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/site_vars.${VARFILE_SUFFIX} 
    ${VARFILE_DIR}/vars.${VARFILE_SUFFIX} 
    DESTINATION etc
)

configure_file(python/tf_config.py.in ${TF_PYPKG_DIR}/tf_config.py @ONLY)
install(FILES ${TF_PYPKG_DIR}/tf_config.py DESTINATION ${TF_SITEPACKAGES}/tissue_forge)
