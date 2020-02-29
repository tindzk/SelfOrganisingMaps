# GCAL models based on Stevens et al. (2013) J. Neurosci.
enable_language(CXX)

add_executable(gcal gcal.cpp)
target_compile_definitions(gcal PUBLIC FLT=float COMPILE_PLOTTING)

# See /usr/lib/cmake/absl/abslTargets.cmake
set(ABSEIL_LIBS absl::flags_parse absl::strings)

target_link_libraries(gcal ${MORPH_LDFLAGS} ${HDF5LIBLINK} ${JSONLIBLINK} ${OpenCV_LIBS} ${ABSEIL_LIBS})
