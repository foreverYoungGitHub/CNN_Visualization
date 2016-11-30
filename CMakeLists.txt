cmake_minimum_required(VERSION 3.6)
project(CNN_Visualization)

add_subdirectory(deps/glfw)

include_directories(deps/glfw/include)
include_directories(deps/glfw/deps)
include_directories(deps/imgui/)
include_directories(deps/imgui/examples/libs/gl3w/)

set(GLAD "deps/glfw/deps/glad/glad.h"
        "deps/glfw/deps/glad.c")

set(gl3w "deps/imgui/examples/libs/gl3w/GL/gl3w.c"
        "deps/imgui/examples/libs/gl3w/GL/gl3w.h"
        "deps/imgui/examples/libs/gl3w/GL/glcorearb.h")

set(imgui "deps/imgui/imgui.h"
        "deps/imgui/imgui.cpp"
        "deps/imgui/imgui_demo.cpp"
        "deps/imgui/imgui_draw.cpp"
        "deps/imgui/imgui_internal.h"
        "deps/imgui/imconfig.h"
        "deps/imgui/stb_rect_pack.h"
        "deps/imgui/stb_textedit.h"
        "deps/imgui/stb_truetype.h")

set(example "deps/imgui/examples/opengl3_example/imgui_impl_glfw_gl3.h"
        "deps/imgui/examples/opengl3_example/imgui_impl_glfw_gl3.cpp"
        "deps/imgui/examples/opengl3_example/main.cpp")

#find_package(OpenGL REQUIRED)

#if (OPENGL_FOUND)
#    message(FATAL_ERROR OPENGL_FOUND${OPENGL_INCLUDE_DIR})
#endif ()

#if (GLEW_FOUND)
#    include_directories(${GLEW_INCLUDE_DIRS})
#    link_libraries(${GLEW_LIBRARIES})
#endif()
#
#if(APPLE)
#    include_directories(/System/Library/Frameworks)
#    find_library(Cocoa_LIBRARY Cocoa)
#    find_library(IOKit_LIBRARY IOKit)
#    find_library(OpenGL_LIBRARY OpenGL)
#    find_path(COREVIDEO_INCLUDE_DIR CoreVideo/CoreVideo.h)
#    MARK_AS_ADVANCED (Cocoa_LIBRARY
#                    IOKit_LIBRARY
#                    OpenGL_LIBRARY)
#    SET(EXTRA_LIBS ${Cocoa_LIBRARY} ${IOKit_LIBRARY} ${OpenGL_LIBRARY})
#ENDIF (APPLE)

#find_package(glfw3 REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)

add_executable(CNN_Visualization ${gl3w} ${imgui} ${example})

target_link_libraries(CNN_Visualization glfw ${GLFW_LIBRARIES})

#target_link_libraries(CNN_Visualization ${EXTRA_LIBS})
#target_include_directories(CNN_Visualization ${OPENGL_INCLUDE_DIR})
#target_link_libraries(CNN_Visualization ${OPENGL_gl_LIBRARY})