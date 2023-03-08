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

# For efficiently aggregating source from lots of models

# This is for initializing processing in a subdirectory
macro(TF_MODEL_TREE_INIT )

  set(TF_MODEL_SRCS_LOCAL )
  set(TF_MODEL_HDRS_LOCAL )

endmacro(TF_MODEL_TREE_INIT)

# This is for posting a source in a subdirectory; path must be absolute or relative to this directory
macro(TF_MODEL_TREE_SRC src_path)

  list(APPEND TF_MODEL_SRCS_LOCAL ${src_path})

endmacro(TF_MODEL_TREE_SRC)

# This is for posting a header in a subdirectory; path must be relative to this directory
macro(TF_MODEL_TREE_HDR hdr_path)

  list(APPEND TF_MODEL_HDRS_LOCAL ${hdr_path})

endmacro(TF_MODEL_TREE_HDR)

# This is for incorporating info from a subdirectory; path must be w.r.t. current source directory
macro(TF_MODEL_TREE_PROC subdir)

  get_directory_property(_TMP 
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}
    DEFINITION TF_MODEL_SRCS_LOCAL
  )
  foreach(_TMPEL ${_TMP})
    TF_MODEL_TREE_SRC(${_TMPEL})
  endforeach(_TMPEL)

  get_directory_property(_TMP 
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}
    DEFINITION TF_MODEL_HDRS_LOCAL
  )
  foreach(_TMPEL ${_TMP})
    TF_MODEL_TREE_HDR(${_TMPEL})
  endforeach(_TMPEL)

endmacro(TF_MODEL_TREE_PROC)
