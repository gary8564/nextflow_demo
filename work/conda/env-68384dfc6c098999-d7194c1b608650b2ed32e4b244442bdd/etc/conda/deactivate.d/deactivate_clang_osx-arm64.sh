#!/bin/bash

# This function takes no arguments
# It tries to determine the name of this file in a programatic way.
function _get_sourced_filename() {
    if [ -n "${BASH_SOURCE[0]}" ]; then
        basename "${BASH_SOURCE[0]}"
    elif [ -n "${(%):-%x}" ]; then
        # in zsh use prompt-style expansion to introspect the same information
        # see http://stackoverflow.com/questions/9901210/bash-source0-equivalent-in-zsh
        basename "${(%):-%x}"
    else
        echo "UNKNOWN FILE"
    fi
}

# The arguments to this are:
# 1. activation nature {activate|deactivate}
# 2. toolchain nature {build|host|ccc}
# 3. machine (should match -dumpmachine)
# 4. prefix (including any final -)
# 5+ program (or environment var comma value)
# The format for 5+ is name{,,value}. If value is specified
#  then name taken to be an environment variable, otherwise
#  it is taken to be a program. In this case, which is used
#  to find the full filename during activation. The original
#  value is stored in environment variable CONDA_BACKUP_NAME
#  For deactivation, the distinction is irrelevant as in all
#  cases NAME simply gets reset to CONDA_BACKUP_NAME.  It is
#  a fatal error if a program is identified but not present.
function _tc_activation() {
  local act_nature=$1; shift
  local tc_prefix=$1; shift
  local thing
  local newval
  local from
  local to
  local pass

  if [ "${act_nature}" = "activate" ]; then
    from=""
    to="CONDA_BACKUP_"
  else
    from="CONDA_BACKUP_"
    to=""
  fi

  for pass in check apply; do
    for thing in "$@"; do
      case "${thing}" in
        *,*)
          newval="${thing#*,}"
          thing="${thing%%,*}"
          ;;
        *)
          newval="${tc_prefix}${thing}"
          thing=$(echo ${thing} | tr 'a-z+-' 'A-ZX_')
          if [ ! -x "${CONDA_PREFIX}/bin/${newval}" -a "${pass}" = "check" ]; then
            echo "ERROR: This cross-compiler package contains no program ${CONDA_PREFIX}/bin/${newval}"
            return 1
          fi
          ;;
      esac
      if [ "${pass}" = "apply" ]; then
        eval oldval="\$${from}$thing"
        if [ -n "${oldval}" ]; then
          eval export "${to}'${thing}'=\"${oldval}\""
        else
          eval unset '${to}${thing}'
        fi
        if [ -n "${newval}" ]; then
          eval export "'${from}${thing}=${newval}'"
        else
          eval unset '${from}${thing}'
        fi
      fi
    done
  done
  return 0
}


function deactivate_clang() {

# When people are using conda-build, assume that adding rpath during build, and pointing at
#    the host env's includes and libs is helpful default behavior
if [ "${CONDA_BUILD:-0}" = "1" ]; then
  CFLAGS_USED="-ftree-vectorize -fPIC -fstack-protector-strong -O2 -pipe -isystem ${PREFIX}/include -fdebug-prefix-map=${SRC_DIR}=/usr/local/src/conda/${PKG_NAME}-${PKG_VERSION} -fdebug-prefix-map=${PREFIX}=/usr/local/src/conda-prefix"
  DEBUG_CFLAGS_USED="-ftree-vectorize -fPIC -fstack-protector-strong -O2 -pipe -Og -g -Wall -Wextra -isystem ${PREFIX}/include -fdebug-prefix-map=${SRC_DIR}=/usr/local/src/conda/${PKG_NAME}-${PKG_VERSION} -fdebug-prefix-map=${PREFIX}=/usr/local/src/conda-prefix"
  LDFLAGS_USED="-Wl,-headerpad_max_install_names -Wl,-dead_strip_dylibs -Wl,-rpath,${PREFIX}/lib -L${PREFIX}/lib"
  LDFLAGS_LD_USED="-headerpad_max_install_names -dead_strip_dylibs -rpath ${PREFIX}/lib -L${PREFIX}/lib"
  CPPFLAGS_USED="-D_FORTIFY_SOURCE=2 -isystem ${PREFIX}/include"
  CMAKE_PREFIX_PATH_USED="${CMAKE_PREFIX_PATH}:${PREFIX}"
else
  CFLAGS_USED="-ftree-vectorize -fPIC -fstack-protector-strong -O2 -pipe -isystem ${CONDA_PREFIX}/include"
  DEBUG_CFLAGS_USED="-ftree-vectorize -fPIC -fstack-protector-strong -O2 -pipe -Og -g -Wall -Wextra -isystem ${CONDA_PREFIX}/include"
  LDFLAGS_USED="-Wl,-headerpad_max_install_names -Wl,-dead_strip_dylibs -Wl,-rpath,${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib"
  LDFLAGS_LD_USED="-headerpad_max_install_names -dead_strip_dylibs -rpath ${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib"
  CPPFLAGS_USED="-D_FORTIFY_SOURCE=2 -isystem ${CONDA_PREFIX}/include"
  CMAKE_PREFIX_PATH_USED="${CMAKE_PREFIX_PATH}:${CONDA_PREFIX}"
fi

if [ "${MACOSX_DEPLOYMENT_TARGET:-0}" != "0" ]; then
  CPPFLAGS_USED="$CPPFLAGS_USED -mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET}"
fi

if [ "${CONDA_BUILD:-0}" = "1" ]; then
  if [ -f /tmp/old-env-$$.txt ]; then
    rm -f /tmp/old-env-$$.txt || true
  fi
  env > /tmp/old-env-$$.txt
fi

CONDA_BUILD_SYSROOT_TEMP=${CONDA_BUILD_SYSROOT:-${SDKROOT:-0}}
if [ "${CONDA_BUILD_SYSROOT_TEMP}" = "0" ]; then
   CONDA_BUILD_SYSROOT_TEMP=$(xcrun --show-sdk-path)
fi

if [ "${CONDA_BUILD:-0}" = "1" ]; then
  # in conda build we need to unset CONDA_BUILD_SYSROOT
  _tc_activation \
    deactivate arm64-apple-darwin20.0.0- \
    "CONDA_BUILD_SYSROOT,${CONDA_BUILD_SYSROOT_TEMP}"
fi

_tc_activation \
  deactivate arm64-apple-darwin20.0.0- "HOST,arm64-apple-darwin20.0.0" \
  "CONDA_TOOLCHAIN_HOST,arm64-apple-darwin20.0.0" \
  "CONDA_TOOLCHAIN_BUILD,arm64-apple-darwin20.0.0" \
  "AR,${AR:-arm64-apple-darwin20.0.0-ar}" \
  "AS,${AS:-arm64-apple-darwin20.0.0-as}" \
  "CHECKSYMS,${CHECKSYMS:-arm64-apple-darwin20.0.0-checksyms}" \
  "INSTALL_NAME_TOOL,${INSTALL_NAME_TOOL:-arm64-apple-darwin20.0.0-install_name_tool}" \
  "LIBTOOL,${LIBTOOL:-arm64-apple-darwin20.0.0-libtool}" \
  "LIPO,${LIPO:-arm64-apple-darwin20.0.0-lipo}" \
  "NM,${NM:-arm64-apple-darwin20.0.0-nm}" \
  "NMEDIT,${NMEDIT:-arm64-apple-darwin20.0.0-nmedit}" \
  "OTOOL,${OTOOL:-arm64-apple-darwin20.0.0-otool}" \
  "PAGESTUFF,${PAGESTUFF:-arm64-apple-darwin20.0.0-pagestuff}" \
  "RANLIB,${RANLIB:-arm64-apple-darwin20.0.0-ranlib}" \
  "REDO_PREBINDING,${REDO_PREBINDING:-arm64-apple-darwin20.0.0-redo_prebinding}" \
  "SEG_ADDR_TABLE,${SEG_ADDR_TABLE:-arm64-apple-darwin20.0.0-seg_addr_table}" \
  "SEG_HACK,${SEG_HACK:-arm64-apple-darwin20.0.0-seg_hack}" \
  "SEGEDIT,${SEGEDIT:-arm64-apple-darwin20.0.0-segedit}" \
  "SIZE,${SIZE:-arm64-apple-darwin20.0.0-size}" \
  "STRINGS,${STRINGS:-arm64-apple-darwin20.0.0-strings}" \
  "STRIP,${STRIP:-arm64-apple-darwin20.0.0-strip}" \
  "CLANG,${CLANG:-arm64-apple-darwin20.0.0-clang}" \
  "LD,${LD:-arm64-apple-darwin20.0.0-ld}" \
  "CC,${CC:-arm64-apple-darwin20.0.0-clang}" \
  "OBJC,${OBJC:-arm64-apple-darwin20.0.0-clang}" \
  "CPP,${CPP:-arm64-apple-darwin20.0.0-clang-cpp}" \
  "CC_FOR_BUILD,${CONDA_PREFIX}/bin/arm64-apple-darwin20.0.0-clang" \
  "OBJC_FOR_BUILD,${CONDA_PREFIX}/bin/@OBJC_FOR_BUILD@" \
  "CPP_FOR_BUILD,${CONDA_PREFIX}/bin/arm64-apple-darwin20.0.0-clang-cpp" \
  "CPPFLAGS,${CPPFLAGS_USED}${CPPFLAGS:+ }${CPPFLAGS:-}" \
  "CFLAGS,${CFLAGS_USED}${CFLAGS:+ }${CFLAGS:-}" \
  "LDFLAGS,${LDFLAGS_USED}${LDFLAGS:+ }${LDFLAGS:-}" \
  "LDFLAGS_LD,${LDFLAGS_LD_USED}${LDFLAGS_LD:+ }${LDFLAGS_LD:-}" \
  "DEBUG_CFLAGS,${DEBUG_CFLAGS_USED}${DEBUG_CFLAGS:+ }${DEBUG_CFLAGS:-}" \
  "_CONDA_PYTHON_SYSCONFIGDATA_NAME,${_CONDA_PYTHON_SYSCONFIGDATA_NAME:-_sysconfigdata_arm64_apple_darwin20_0_0}" \
  "CMAKE_PREFIX_PATH,${CMAKE_PREFIX_PATH:-${CMAKE_PREFIX_PATH_USED}}" \
  "CONDA_BUILD_CROSS_COMPILATION," \
  "SDKROOT,${CONDA_BUILD_SYSROOT_TEMP}" \
  "CMAKE_ARGS,${_CMAKE_ARGS:-}" \
  "MESON_ARGS,${_MESON_ARGS:-}" \
  "ac_cv_func_malloc_0_nonnull,yes" \
  "ac_cv_func_realloc_0_nonnull,yes" \
  "host_alias,arm64-apple-darwin20.0.0" \
  "build_alias,arm64-apple-darwin20.0.0" \
  "BUILD,arm64-apple-darwin20.0.0"

unset CONDA_BUILD_SYSROOT_TEMP

if [ $? -ne 0 ]; then
  echo "ERROR: $(_get_sourced_filename) failed, see above for details"
else
  if [ "${CONDA_BUILD:-0}" = "1" ]; then
    if [ -f /tmp/new-env-$$.txt ]; then
      rm -f /tmp/new-env-$$.txt || true
    fi
    env > /tmp/new-env-$$.txt

    echo "INFO: $(_get_sourced_filename) made the following environmental changes:"
    diff -U 0 -rN /tmp/old-env-$$.txt /tmp/new-env-$$.txt | tail -n +4 | grep "^-.*\|^+.*" | grep -v "CONDA_BACKUP_" | sort
    rm -f /tmp/old-env-$$.txt /tmp/new-env-$$.txt || true
  fi

  # unfix prompt for zsh
  if [[ -n "${ZSH_NAME:-}" ]]; then
    precmd_functions=(${precmd_functions:#_conda_clang_precmd})
    preexec_functions=(${preexec_functions:#_conda_clang_preexec})
  fi
fi
}

if [ "${CONDA_BUILD_STATE:-0}" = "BUILD" ] && [ "${target_platform:-osx-arm64}" != "osx-arm64" ]; then
  echo "Not deactivating environment because this compiler is not expected."
else
  deactivate_clang
fi
