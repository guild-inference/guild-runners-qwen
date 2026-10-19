#include <guild/abi/runner_abi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>
extern "C" guild_status_t guild_plugin_entry_v1(uint32_t,guild_plugin_descriptor_v1_t*,guild_error_info_t*);
#define CHECK(x) do { if(!(x)) { std::fprintf(stderr,"failed %s:%d: %s\n",__FILE__,__LINE__,#x); std::abort(); } } while(0)

struct MockModel {
    std::string arch = "qwen35moe";
    std::map<std::string, int64_t> i64_keys{
        {"qwen35moe.block_count", 40},
        {"qwen35moe.embedding_length", 2048},
        {"qwen35moe.expert_count", 256},
        {"qwen35moe.expert_used_count", 8},
        {"qwen35moe.attention.head_count", 16},
        {"qwen35moe.attention.head_count_kv", 2},
    };
    int64_t ple_dim0 = 160;
    bool has_ple_tensor = true;
};

static guild_model_reader_v1_t make_reader(MockModel& mm) {
    guild_model_reader_v1_t reader{};
    reader.struct_size = sizeof(reader);
    reader.user = &mm;
    reader.get_string = [](void* u, const char* k, char* out, size_t n) -> guild_status_t {
        const auto* m = static_cast<const MockModel*>(u);
        if (std::strcmp(k, "general.architecture") == 0) {
            if (m->arch.empty()) return GUILD_STATUS_ERR_INVALID_ARGUMENT;
            std::snprintf(out, n, "%s", m->arch.c_str());
            return GUILD_STATUS_OK;
        }
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    };
    reader.get_i64 = [](void* u, const char* k, int64_t* out) -> guild_status_t {
        const auto* m = static_cast<const MockModel*>(u);
        const auto it = m->i64_keys.find(k);
        if (it == m->i64_keys.end()) return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        *out = it->second;
        return GUILD_STATUS_OK;
    };
    reader.get_f64 = [](void*, const char*, double*) -> guild_status_t {
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    };
    reader.get_tensor = [](void* u, const char* name, guild_tensor_desc_t* t) -> guild_status_t {
        const auto* m = static_cast<const MockModel*>(u);
        if (std::strcmp(name, "per_layer_token_embd.weight") == 0) {
            if (!m->has_ple_tensor) return GUILD_STATUS_ERR_INVALID_TENSOR;
            t->n_dims = 2;
            t->dims[0] = m->ple_dim0;
            t->dims[1] = 20000096 * 16;
            return GUILD_STATUS_OK;
        }
        const int64_t embd = (m->arch == "qwen4exp" || m->i64_keys.count("qwen4exp.embedding_length")) ? 2560 : 2048;
        t->n_dims = 2;
        t->dims[0] = embd;
        t->dims[1] = 248320;
        return GUILD_STATUS_OK;
    };
    return reader;
}

static int count=0;
static guild_status_t collect(guild_core_host_t,const guild_runner_desc_v1_t* d,guild_error_info_t*) {
    CHECK(d->next_ext && d->package_id && d->n_required_op_kinds);
    const auto* ext=static_cast<const guild_runner_model_ext_v1_t*>(d->next_ext);
    CHECK(ext->struct_size==sizeof(*ext) && ext->priority>0);
    const bool is_qwen4 = (std::strcmp(d->arch_id, "qwen4exp") == 0);
    CHECK(d->n_required_op_kinds == (is_qwen4 ? 23u : 19u));
    if (!is_qwen4) {
        CHECK((d->required_core_caps & GUILD_CORE_CAP_BATCHED_PREFILL) != 0);
    }

    MockModel mm35;
    auto reader35 = make_reader(mm35);
    uint32_t yes = 0;
    CHECK(ext->recognize(&reader35, &yes) == GUILD_STATUS_OK);
    if (std::strcmp(d->arch_id, "qwen35moe") == 0) {
        CHECK(yes == 1);
    } else if (is_qwen4) {
        CHECK(yes == 0);
    }

    guild_runner_model_v1_t m{};
    m.struct_size=sizeof(m);
    CHECK(ext->describe(&reader35,&m,nullptr)==GUILD_STATUS_OK);
    CHECK(m.residual_streams==1 && m.n_layers==40 && m.ssm_v_heads==32 && m.active_experts==8);
    CHECK(m.prefill_strategy==GUILD_RUNNER_PREFILL_LAYER_BATCHED && !m.requires_ple);
    m.struct_size=4;
    CHECK(ext->describe(&reader35,&m,nullptr)==GUILD_STATUS_ERR_ABI_MISMATCH);

    // Missing mandatory key must fail closed
    MockModel mm_missing = mm35;
    mm_missing.i64_keys.erase("qwen35moe.block_count");
    auto reader_missing = make_reader(mm_missing);
    m.struct_size = sizeof(m);
    guild_error_info_t err{};
    CHECK(ext->describe(&reader_missing, &m, &err) == GUILD_STATUS_ERR_INVALID_TENSOR);
    CHECK(std::strstr(err.message, "qwen35moe.block_count") != nullptr);

    // Fallback recognition when general.architecture is absent
    MockModel mm_fallback = mm35;
    mm_fallback.arch.clear();
    auto reader_fallback = make_reader(mm_fallback);
    if (std::strcmp(d->arch_id, "qwen35moe") == 0) {
        CHECK(ext->recognize(&reader_fallback, &yes) == GUILD_STATUS_OK && yes == 1);
        CHECK(ext->describe(&reader_fallback, &m, nullptr) == GUILD_STATUS_OK);
    }

    // Qwen4Exp validation: canonical dimensions, PLE metadata, and PLE tensor shape
    if (is_qwen4) {
        MockModel mm4;
        mm4.arch = "qwen4exp";
        mm4.i64_keys = {
            {"qwen4exp.block_count", 48},
            {"qwen4exp.embedding_length", 2560},
            {"qwen4exp.expert_count", 512},
            {"qwen4exp.expert_used_count", 10},
            {"qwen4exp.attention.head_count", 24},
            {"qwen4exp.attention.head_count_kv", 2},
            {"qwen4exp.ple.vocab_per_layer", 20000096},
            {"qwen4exp.ple.embedding_length", 2560},
            {"qwen4exp.ple.key_length", 256},
            {"qwen4exp.ple.ngram_min", 2},
            {"qwen4exp.ple.ngram_max", 3},
        };
        auto reader4 = make_reader(mm4);
        CHECK(ext->recognize(&reader4, &yes) == GUILD_STATUS_OK && yes == 1);
        CHECK(ext->describe(&reader4, &m, nullptr) == GUILD_STATUS_OK);
        CHECK(m.residual_streams == 4 && m.n_layers == 48 && m.requires_ple == 1);

        // Wrong PLE tensor shape must fail closed
        mm4.ple_dim0 = 128;
        CHECK(ext->describe(&reader4, &m, &err) == GUILD_STATUS_ERR_INVALID_TENSOR);
        mm4.ple_dim0 = 160;

        // Invalid PLE metadata must fail closed
        mm4.i64_keys["qwen4exp.ple.ngram_max"] = 4;
        CHECK(ext->describe(&reader4, &m, &err) == GUILD_STATUS_ERR_INVALID_TENSOR);
        mm4.i64_keys.erase("qwen4exp.ple.vocab_per_layer");
        mm4.i64_keys.erase("qwen4exp.ple.embedding_length");
        mm4.i64_keys.erase("qwen4exp.ple.key_length");
        mm4.i64_keys.erase("qwen4exp.ple.ngram_min");
        mm4.i64_keys.erase("qwen4exp.ple.ngram_max");

        // Invalid PLE artifact metadata must fail closed
        mm4.i64_keys["qwen4exp.ple.ngram_size"] = 3;
        mm4.i64_keys["qwen4exp.ple.heads_per_ngram"] = 8;
        mm4.i64_keys["qwen4exp.ple.conv_kernel"] = 4;
        mm4.i64_keys["qwen4exp.ple.eos_token_id"] = 999;
        CHECK(ext->describe(&reader4, &m, &err) == GUILD_STATUS_ERR_INVALID_TENSOR);
        mm4.i64_keys["qwen4exp.ple.eos_token_id"] = 248044;
        CHECK(ext->describe(&reader4, &m, nullptr) == GUILD_STATUS_OK);
    }

    // Vectorized finite validation across 64-element blocks and tail
    CHECK(ext->validate_numerical(nullptr, 0, &err) == GUILD_STATUS_ERR_INVALID_ARGUMENT);
    std::vector<float> vals(150, 1.25f);
    CHECK(ext->validate_numerical(vals.data(), vals.size(), &err) == GUILD_STATUS_OK);
    vals[63] = std::numeric_limits<float>::quiet_NaN();
    CHECK(ext->validate_numerical(vals.data(), vals.size(), &err) == GUILD_STATUS_ERR_NONFINITE_OUTPUT);
    CHECK(std::strstr(err.message, "63") != nullptr);
    vals[63] = 1.25f;
    vals[149] = std::numeric_limits<float>::infinity();
    CHECK(ext->validate_numerical(vals.data(), vals.size(), &err) == GUILD_STATUS_ERR_NONFINITE_OUTPUT);
    CHECK(std::strstr(err.message, "149") != nullptr);

    ++count;return GUILD_STATUS_OK;
}
int main() {
    guild_plugin_descriptor_v1_t d{};
    CHECK(guild_plugin_entry_v1(GUILD_KERNEL_ABI_VERSION_V1,&d,nullptr)==GUILD_STATUS_OK);
    guild_host_api_v1_t host{};host.struct_size=sizeof(host);host.register_runner=collect;
    CHECK(d.init(nullptr,&host,nullptr)==GUILD_STATUS_OK && count==4);
    CHECK(guild_plugin_entry_v1(GUILD_MAKE_ABI_VERSION(2,0,0),&d,nullptr)==GUILD_STATUS_ERR_ABI_MISMATCH);
}
