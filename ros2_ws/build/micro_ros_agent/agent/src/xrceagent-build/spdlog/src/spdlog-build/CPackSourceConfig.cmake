# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BUILD_SOURCE_DIRS "/home/cadt-02/Object-Detection/ros2_ws/build/micro_ros_agent/agent/src/xrceagent-build/spdlog/src/spdlog;/home/cadt-02/Object-Detection/ros2_ws/build/micro_ros_agent/agent/src/xrceagent-build/spdlog/src/spdlog-build")
set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEBIAN_FILE_NAME "spdlog-1.9.2.deb")
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Very fast, header-only/compiled, C++ logging library.")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/gabime/spdlog")
set(CPACK_DEBIAN_PACKAGE_NAME "spdlog")
set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "/home/cadt-02/.local/lib/python3.8/site-packages/cmake/data/share/cmake-3.28/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "spdlog built using CMake")
set(CPACK_GENERATOR "TBZ2;TGZ;TXZ;TZ")
set(CPACK_GENERATOR "TBZ2;TGZ;TXZ;TZ")
set(CPACK_IGNORE_FILES "/CVS/;/\\.svn/;/\\.bzr/;/\\.hg/;/\\.git/;\\.swp\$;\\.#;/#")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY "0")
set(CPACK_INNOSETUP_ARCHITECTURE "x64")
set(CPACK_INSTALLED_DIRECTORIES "/home/cadt-02/Object-Detection/ros2_ws/build/micro_ros_agent/agent/src/xrceagent-build/spdlog/src/spdlog;/")
set(CPACK_INSTALL_CMAKE_PROJECTS "")
set(CPACK_INSTALL_PREFIX "/home/cadt-02/Object-Detection/ros2_ws/build/micro_ros_agent/agent/src/xrceagent-build/temp_install/spdlog-1.9.2")
set(CPACK_MODULE_PATH "")
set(CPACK_NSIS_DISPLAY_NAME "spdlog 1.9.2")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
set(CPACK_NSIS_PACKAGE_NAME "spdlog 1.9.2")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_OBJCOPY_EXECUTABLE "/usr/bin/objcopy")
set(CPACK_OBJDUMP_EXECUTABLE "/usr/bin/objdump")
set(CPACK_OUTPUT_CONFIG_FILE "/home/cadt-02/Object-Detection/ros2_ws/build/micro_ros_agent/agent/src/xrceagent-build/spdlog/src/spdlog-build/CPackConfig.cmake")
set(CPACK_PACKAGE_CONTACT "Gabi Melman <gmelman1@gmail.com>")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "/home/cadt-02/.local/lib/python3.8/site-packages/cmake/data/share/cmake-3.28/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Fast C++ logging library")
set(CPACK_PACKAGE_FILE_NAME "spdlog-1.9.2-Source")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "spdlog 1.9.2")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "spdlog 1.9.2")
set(CPACK_PACKAGE_NAME "spdlog")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "Gabi Melman")
set(CPACK_PACKAGE_VERSION "1.9.2")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "9")
set(CPACK_PACKAGE_VERSION_PATCH "2")
set(CPACK_PROJECT_URL "https://github.com/gabime/spdlog")
set(CPACK_READELF_EXECUTABLE "/usr/bin/readelf")
set(CPACK_RESOURCE_FILE_LICENSE "/home/cadt-02/.local/lib/python3.8/site-packages/cmake/data/share/cmake-3.28/Templates/CPack.GenericLicense.txt")
set(CPACK_RESOURCE_FILE_README "/home/cadt-02/.local/lib/python3.8/site-packages/cmake/data/share/cmake-3.28/Templates/CPack.GenericDescription.txt")
set(CPACK_RESOURCE_FILE_WELCOME "/home/cadt-02/.local/lib/python3.8/site-packages/cmake/data/share/cmake-3.28/Templates/CPack.GenericWelcome.txt")
set(CPACK_RPM_FILE_NAME "spdlog-1.9.2.rpm")
set(CPACK_RPM_PACKAGE_DESCRIPTION "Very fast, header-only/compiled, C++ logging library.")
set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_NAME "spdlog")
set(CPACK_RPM_PACKAGE_SOURCES "ON")
set(CPACK_RPM_PACKAGE_URL "https://github.com/gabime/spdlog")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_GENERATOR "TBZ2;TGZ;TXZ;TZ")
set(CPACK_SOURCE_IGNORE_FILES "/CVS/;/\\.svn/;/\\.bzr/;/\\.hg/;/\\.git/;\\.swp\$;\\.#;/#")
set(CPACK_SOURCE_INSTALLED_DIRECTORIES "/home/cadt-02/Object-Detection/ros2_ws/build/micro_ros_agent/agent/src/xrceagent-build/spdlog/src/spdlog;/")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/home/cadt-02/Object-Detection/ros2_ws/build/micro_ros_agent/agent/src/xrceagent-build/spdlog/src/spdlog-build/CPackSourceConfig.cmake")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "spdlog-1.9.2-Source")
set(CPACK_SOURCE_RPM "OFF")
set(CPACK_SOURCE_TBZ2 "ON")
set(CPACK_SOURCE_TGZ "ON")
set(CPACK_SOURCE_TOPLEVEL_TAG "Linux-Source")
set(CPACK_SOURCE_TXZ "ON")
set(CPACK_SOURCE_TZ "ON")
set(CPACK_SOURCE_ZIP "OFF")
set(CPACK_STRIP_FILES "")
set(CPACK_SYSTEM_NAME "Linux")
set(CPACK_THREADS "1")
set(CPACK_TOPLEVEL_TAG "Linux-Source")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "/home/cadt-02/Object-Detection/ros2_ws/build/micro_ros_agent/agent/src/xrceagent-build/spdlog/src/spdlog-build/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()
