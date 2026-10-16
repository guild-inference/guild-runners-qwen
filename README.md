# guild-runners-qwen (Phase 2 migration preview)

Independent Qwen model-description plugin, built against the installed **Guild
Kernel SDK 1.1**. No Guild source/build directory is a build dependency.

The plugin supplies GGUF recognition, descriptor construction, hybrid geometry,
canonical tensor roles, PLE requirements, prefill selection and required operation
kinds through size-checked C callbacks. Core owns weights, sessions, scheduling
and graph execution. Qwen35MoE/Ornith inference has been verified against Core's
built-in runner, including bitwise first-token logits, eight greedy IDs, recurrent
state and the tested host-only KV prefix. Core's built-in runner registration can
be disabled with `GUILD_BUILTIN_QWEN_RUNNERS=OFF`.

**Scope still in migration:** Qwen4Exp descriptors build but real Qwen4Exp
inference has not been verified with this plugin. Layer tensor validation, the
execution graph and some architecture policy still reside in Core. Arbitrary
tensor naming and non-periodic topology are not supported by this contract.
This repository does not establish complete architecture independence yet.

## Build and test

Install SDK 1.1 (a header-only package), then:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/guild-sdk
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/build/package"
```

Tests retain checks under `NDEBUG`. C99 header compilation and a synthetic model
reader verify the public contract. The real-model integration driver lives in
Guild as `tools/test_external_runner.py`; it takes separately installed package
directories, checks external provider IDs and performs fresh-process inference.

## Sign and install

Use a private Ed25519 seed file (64 hex characters, owner-readable only) and a
public key distributed through your own trusted channel. Core embeds no private
release signing identity. Never commit the seed file.

```sh
guild kernel sign build/package --key-id YOUR-KEY-ID --seed-file /secure/release.seed --bundle build/qwen-1.0.0.gkernel
guild kernel trust-key --publisher org.guild-inference --key-id YOUR-KEY-ID --pubkey YOUR-PUBLIC-KEY
guild kernel install build/qwen-1.0.0.gkernel
guild kernel active --json
```

The manifest contains artifact hashes recomputed by the signer. Package ID:
`org.guild.runners.qwen`; version `1.0.0`; Kernel ABI major 1. The optional
`RMD1` extension requires SDK/Core support introduced in Guild `4b4ded0`.
Built-in runners remain the recovery path when this package is disabled.

Source provenance: Qwen runner/archetype behavior from Guild `42018db`, adapted
to the public C contract. MIT, including the original Strata attribution in
`LICENSE`. No generic hardware kernels are duplicated here.
