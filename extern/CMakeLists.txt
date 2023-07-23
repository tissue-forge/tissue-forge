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

# Fixing seeming libxml2 bug not finding unicode headers on osx
if(TF_APPLE)
  include_directories(${CMAKE_FIND_ROOT_PATH}/include)
endif()

set(WITH_PNGIMPORTER OFF CACHE BOOL "build PNG importer"    FORCE)

# magnum / corrade want this
# stop messaging about global weak symbols.
set(BUILD_STATIC_UNIQUE_GLOBALS OFF CACHE BOOL "Do something with symbols to make them work on Windows" FORCE)

add_subdirectory(corrade)

# set magnum options
set(WITH_GLFWAPPLICATION ON CACHE BOOL "build magnum with glfw" FORCE)
set(BUILD_STATIC ON CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_PLUGINS_STATIC ON CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_TESTS OFF CACHE BOOL "build as static libraries for now..." FORCE)
set(BUILD_GL_TESTS OFF CACHE BOOL "build as static libraries for now..." FORCE)

# magnum plugins for mesh import testing
set(WITH_ASSIMPIMPORTER    OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_OBJIMPORTER       OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_OPENGEXIMPORTER   OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_ANYIMAGECONVERTER OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_AUDIO             OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_ANYSCENEIMPORTER  OFF CACHE BOOL "build Assimp importer" FORCE)
set(WITH_PNGIMPORTER       OFF CACHE BOOL "build PNG importer"    FORCE)

if(APPLE)
  set(WITH_CGLCONTEXT ON CACHE BOOL "Build CglContext library" FORCE)

  set(WITH_WINDOWLESSCGLAPPLICATION ON CACHE BOOL "Build WindowlessCglApplication library" FORCE)

elseif(UNIX AND NOT APPLE)
  set(WITH_WINDOWLESSEGLAPPLICATION ON CACHE BOOL "Build WindowlessEglApplication library" FORCE)

  set(WITH_WINDOWLESSGLXAPPLICATION ON CACHE BOOL "Build WindowlessGlxApplication library" FORCE)

elseif(WIN32)
  set(WITH_WINDOWLESSWGLAPPLICATION ON CACHE BOOL "Build WindowlessWglApplication library" FORCE)

endif()

set(WITH_TGAIMAGECONVERTER ON CACHE BOOL "Build TGA Image Converter" FORCE)

add_subdirectory(magnum)

set(WITH_STBIMAGECONVERTER ON CACHE BOOL "Build Stb Image Converter" FORCE)

add_subdirectory(magnum-plugins EXCLUDE_FROM_ALL)

# build roadrunner

# make sure to correctly target roadrunner dependencies
set(CMAKE_PREFIX_PATH 
  ${TF_ROADRUNNER_ROOT}
  ${TF_ROADRUNNER_ROOT}/lib/cmake
  ${CMAKE_PREFIX_PATH}
)

set(RR_DEPENDENCIES_INSTALL_PREFIX ${TF_PRIVATE_INSTALL_DIR}/roadrunner CACHE PATH "Path to dependency package" FORCE)

add_subdirectory(roadrunner)
