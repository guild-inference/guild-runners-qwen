# Canonical Qwen model-runner source attachment for bundled and standalone builds.
get_filename_component(GUILD_QWEN_RUNNERS_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(GUILD_QWEN_RUNNERS_SOURCES "${GUILD_QWEN_RUNNERS_ROOT}/src/qwen.cpp")
set(GUILD_QWEN_RUNNERS_INCLUDE_DIRS "${GUILD_QWEN_RUNNERS_ROOT}")
function(guild_qwen_runners_attach target)
  target_sources(${target} PRIVATE "${GUILD_QWEN_RUNNERS_ROOT}/src/qwen.cpp")
  set_source_files_properties("${GUILD_QWEN_RUNNERS_ROOT}/src/qwen.cpp"
    PROPERTIES COMPILE_DEFINITIONS "GUILD_BUNDLED_QWEN_RUNNERS=1")
endfunction()
