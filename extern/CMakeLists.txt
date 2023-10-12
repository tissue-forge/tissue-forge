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

# build libsbml

set(LIBSBML_SHARED_VERSION OFF CACHE BOOL "dont build shared libsbml")

set(WITH_CPP_NAMESPACE ON CACHE BOOL "Use a C++ namespace for libSBML.")

set(WITH_PYTHON OFF CACHE BOOL "Generate the Python language interface for libSBML." )

set(WITH_SWIG OFF CACHE BOOL "Regenerate the programming language interface code (for Java, Python, etc.) using SWIG.")

set(LIBSBML_ROOT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libsbml" CACHE PATH "Path to the libSBML root source directory" FORCE)

set(LIBSBML_ROOT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/libsbml" CACHE PATH "Path to the libSBML root build directory" FORCE)

set(LIBSBML_SKIP_SHARED_LIBRARY "ON")

# If necessary, confine doc files to share, rather than in install root
if(NOT UNIX AND NOT CYGWIN)
  function(libsbml_doc_change variable access value current stack)
    set(MISC_PREFIX "./share/sbml" PARENT_SCOPE)
  endfunction()

  variable_watch(MISC_PREFIX libsbml_doc_change)
endif()

add_subdirectory(libsbml)

# create an interface target for libsbml as a modern cmake target

add_library(libsbml-static-interface INTERFACE)

# libsbml names libs differently, why???
if(UNIX OR CYGWIN)
  set(LIBSBML_LIBRARY sbml)
else()
  if(MINGW)
    set(LIBSBML_LIBRARY sbml)
  else()
    set(LIBSBML_LIBRARY libsbml)
  endif()
endif()

set_property(TARGET ${LIBSBML_LIBRARY}-static PROPERTY C_STANDARD 11)
set_property(TARGET ${LIBSBML_LIBRARY}-static PROPERTY CXX_STANDARD 14)

target_link_libraries(libsbml-static-interface INTERFACE ${LIBSBML_LIBRARY}-static)

target_compile_definitions(libsbml-static-interface INTERFACE LIBSBML_STATIC)

target_include_directories(libsbml-static-interface INTERFACE "$<TARGET_PROPERTY:${LIBSBML_LIBRARY}-static,INCLUDE_DIRECTORIES>")

add_library(LibSBML::LibSBML ALIAS libsbml-static-interface)

install(
  TARGETS libsbml-static-interface 
  EXPORT libsbml-static-interface_targets 
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" 
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" 
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(
  EXPORT libsbml-static-interface_targets 
  FILE TissueForge-libSBMLTargets.cmake 
  NAMESPACE TissueForge:: 
  DESTINATION ${TF_SHARE_DIR} 
  EXPORT_LINK_INTERFACE_LIBRARIES 
  COMPONENT libSBML
)

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
set(WITH_STBTRUETYPEFONT ON CACHE BOOL "Build Stb True Type Font" FORCE)

add_subdirectory(magnum-plugins EXCLUDE_FROM_ALL)

set(WITH_UI ON CACHE BOOL "Ui support" FORCE)

add_subdirectory(magnum-extras EXCLUD_FROM_ALL)
