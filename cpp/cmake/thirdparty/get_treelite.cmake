#=============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# function(find_treelite_in_python_wheel PKG_VERSION PKG_PINNED_TAG)

#   function(find_treelite_lib _name _dir _bin _src)
#     if(_dir)
#       list(APPEND CMAKE_PREFIX_PATH "${_dir}")
#     endif()
#     rapids_find_generate_module(
#       ${_name}
#       NO_CONFIG
#       VERSION ${PKG_VERSION}
#       BUILD_EXPORT_SET cuml-exports
#       INSTALL_EXPORT_SET cuml-exports
#       HEADER_NAMES "__init__.py"
#       LIBRARY_NAMES "${_name}"
#     )
#     find_package(${_name} MODULE REQUIRED GLOBAL)
#     if(NOT TARGET treelite::${_name})
#       add_library(treelite::${_name} ALIAS ${_name})
#     endif()
#     # clear out INTERFACE_INCLUDE_DIRECTORIES
#     foreach(_target IN ITEMS "${_name}::${_name}")
#       get_target_property(_val ${_target} INTERFACE_INCLUDE_DIRECTORIES)
#       set_target_properties(${_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
#       message(VERBOSE "find_treelite_lib ${_target} INTERFACE_INCLUDE_DIRECTORIES is '${_val}'")
#     endforeach()
#     # Set INTERFACE_INCLUDE_DIRECTORIES to the downloaded binary/source include dirs
#     target_include_directories(${_name}
#                      INTERFACE $<BUILD_INTERFACE:${_bin}/include>
#                                $<BUILD_INTERFACE:${_src}/include>
#                                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
#   endfunction()

#   CPMAddPackage(
#     NAME           Treelite
#     VERSION        ${PKG_VERSION}
#     GIT_REPOSITORY https://github.com/dmlc/treelite.git
#     GIT_TAG        ${PKG_PINNED_TAG}
#     DOWNLOAD_ONLY
#   )

#   install(DIRECTORY ${Treelite_BINARY_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#   install(DIRECTORY ${Treelite_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#   find_treelite_lib(treelite
#     "${CUML_TREELITE_WHEEL_DIR}"
#     "${Treelite_BINARY_DIR}" "${Treelite_SOURCE_DIR}")
#   find_treelite_lib(treelite_runtime
#     "${CUML_TREELITE_RUNTIME_WHEEL_DIR}"
#     "${Treelite_BINARY_DIR}" "${Treelite_SOURCE_DIR}")

#   set(TREELITE_LIBS treelite::treelite treelite::treelite_runtime PARENT_SCOPE)
# endfunction()

function(find_and_configure_treelite)

    set(oneValueArgs VERSION PINNED_TAG EXCLUDE_FROM_ALL BUILD_STATIC_LIBS)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    # if(CUML_LINK_TO_TREELITE_WHEEL AND (NOT PKG_BUILD_STATIC_LIBS))
    #   find_treelite_in_python_wheel(${PKG_VERSION} ${PKG_PINNED_TAG})
    #   set(TREELITE_LIBS ${TREELITE_LIBS} PARENT_SCOPE)
    #   return()
    # endif()

    if(NOT PKG_BUILD_STATIC_LIBS)
      list(APPEND TREELITE_LIBS treelite::treelite treelite::treelite_runtime)
    else()
      list(APPEND TREELITE_LIBS treelite::treelite_static treelite::treelite_runtime_static)
    endif()

    rapids_cpm_find(Treelite ${PKG_VERSION}
        GLOBAL_TARGETS       ${TREELITE_LIBS}
        INSTALL_EXPORT_SET   cuml-exports
        CPM_ARGS
            GIT_REPOSITORY   https://github.com/dmlc/treelite.git
            GIT_TAG          ${PKG_PINNED_TAG}
            EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
            OPTIONS
              "USE_OPENMP ON"
              "BUILD_STATIC_LIBS ${PKG_BUILD_STATIC_LIBS}"
    )


    list(APPEND TREELITE_LIBS_NO_PREFIX treelite treelite_runtime)
    if(Treelite_ADDED AND PKG_BUILD_STATIC_LIBS)
        list(APPEND TREELITE_LIBS_NO_PREFIX treelite_static treelite_runtime_static)
    endif()

    set(Treelite_ADDED ${Treelite_ADDED} PARENT_SCOPE)
    set(TREELITE_LIBS ${TREELITE_LIBS} PARENT_SCOPE)
    if(Treelite_ADDED)
        if (NOT PKG_BUILD_STATIC_LIBS)
            target_include_directories(treelite
                PUBLIC $<BUILD_INTERFACE:${Treelite_SOURCE_DIR}/include>
                       $<BUILD_INTERFACE:${Treelite_BINARY_DIR}/include>)
            target_include_directories(treelite_runtime
                PUBLIC $<BUILD_INTERFACE:${Treelite_SOURCE_DIR}/include>
                       $<BUILD_INTERFACE:${Treelite_BINARY_DIR}/include>)
            if(NOT TARGET treelite::treelite)
                add_library(treelite::treelite ALIAS treelite)
            endif()
            if(NOT TARGET treelite::treelite_runtime)
                add_library(treelite::treelite_runtime ALIAS treelite_runtime)
            endif()
        else()
            target_include_directories(treelite_static
                PUBLIC $<BUILD_INTERFACE:${Treelite_SOURCE_DIR}/include>
                       $<BUILD_INTERFACE:${Treelite_BINARY_DIR}/include>)
            target_include_directories(treelite_runtime_static
                PUBLIC $<BUILD_INTERFACE:${Treelite_SOURCE_DIR}/include>
                       $<BUILD_INTERFACE:${Treelite_BINARY_DIR}/include>)
            if(NOT TARGET treelite::treelite_static)
                add_library(treelite::treelite_static ALIAS treelite_static)
            endif()
            if(NOT TARGET treelite::treelite_runtime_static)
                add_library(treelite::treelite_runtime_static ALIAS treelite_runtime_static)
            endif()
        endif()

        rapids_export(BUILD Treelite
            EXPORT_SET TreeliteTargets
            GLOBAL_TARGETS ${TREELITE_LIBS_NO_PREFIX}
            NAMESPACE treelite::)
    endif()

    # We generate the treelite-config files when we built treelite locally, so always do `find_dependency`
    rapids_export_package(BUILD Treelite cuml-exports)

    # Tell cmake where it can find the generated treelite-config.cmake we wrote.
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD Treelite [=[${CMAKE_CURRENT_LIST_DIR}]=] cuml-exports)
endfunction()

find_and_configure_treelite(VERSION     2.4.0
                            PINNED_TAG  dcd54779ce9324c69452db9b906f2d258374d5b9
                            EXCLUDE_FROM_ALL  ${CUML_EXCLUDE_TREELITE_FROM_ALL}
                            BUILD_STATIC_LIBS ${CUML_USE_TREELITE_STATIC})
