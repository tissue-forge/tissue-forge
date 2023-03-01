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

function(prepend var)
  set(listVar "")
  foreach(f ${ARGN})
    list(APPEND listVar "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
  endforeach(f)
  set(${var} "${listVar}" PARENT_SCOPE)
endfunction(prepend)

configure_file(tf_config.h.in tf_config.h)

set(TF_INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(_PUBLIC_HEADERS_STATIC
    TissueForge.h
    tf_port.h
    tf_runtime.h
    tf_style.h
)

set(_PUBLIC_HEADERS
    ${_PUBLIC_HEADERS_STATIC}
    tf_config.h
)

prepend(PUBLIC_HEADERS  ${_PUBLIC_HEADERS})

set(TISSUEFORGE_PUBLIC_HEADERS_STATIC ${_PUBLIC_HEADERS_STATIC} PARENT_SCOPE)

set(TISSUEFORGE_PUBLIC_HEADERS ${PUBLIC_HEADERS} PARENT_SCOPE)

set(TF_INCLUDE_DIRS ${TF_INCLUDE_DIRS} PARENT_SCOPE)

install(FILES 
        ${_PUBLIC_HEADERS_STATIC} 
        ${CMAKE_CURRENT_BINARY_DIR}/tf_config.h 
        DESTINATION include/TissueForge
)
