// Copyright (c) Guild Inference Contributors. MIT License.
// Architecture contracts extracted from Guild 42018db, runner.cpp/archetype.cpp.
#include <guild/abi/runner_abi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#ifndef GUILD_PACKAGE_VERSION
#define GUILD_PACKAGE_VERSION "1.1.0"
#endif
namespace {
int64_t integer(const guild_model_reader_v1_t* r, const std::string& key, int64_t fallback) {
    int64_t value=0;
    return r->get_i64(r->user,key.c_str(),&value)==GUILD_STATUS_OK ? value : fallback;
}
bool has_integer(const guild_model_reader_v1_t* r, const std::string& key, int64_t* out = nullptr) {
    int64_t value=0;
    if (!r || !r->get_i64 || r->get_i64(r->user,key.c_str(),&value)!=GUILD_STATUS_OK) return false;
    if (out) *out = value;
    return true;
}
double number(const guild_model_reader_v1_t* r, const std::string& key, double fallback) {
    double value=0;
    return r->get_f64(r->user,key.c_str(),&value)==GUILD_STATUS_OK ? value : fallback;
}
std::string text(const guild_model_reader_v1_t* r, const char* key, const char* fallback) {
    char value[256]={};
    return r->get_string(r->user,key,value,sizeof(value))==GUILD_STATUS_OK ? value : fallback;
}
std::string lower_text(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c); });
    return s;
}
guild_status_t recognize_qwen4exp(const guild_model_reader_v1_t* r, uint32_t* yes) {
    if (!r || r->struct_size<sizeof(*r) || !yes || !r->get_string) return GUILD_STATUS_ERR_ABI_MISMATCH;
    const auto arch=lower_text(text(r,"general.architecture",""));
    if (!arch.empty()) {
        *yes = (arch=="qwen4exp" || arch=="qwen3.8-flash-next") ? 1u : 0u;
        return GUILD_STATUS_OK;
    }
    *yes = has_integer(r,"qwen4exp.expert_count") ? 1u : 0u;
    return GUILD_STATUS_OK;
}
guild_status_t recognize_qwen35moe(const guild_model_reader_v1_t* r, uint32_t* yes) {
    if (!r || r->struct_size<sizeof(*r) || !yes || !r->get_string) return GUILD_STATUS_ERR_ABI_MISMATCH;
    const auto arch=lower_text(text(r,"general.architecture",""));
    if (!arch.empty()) {
        *yes = (arch=="qwen35moe" || arch=="qwen2moe" || arch=="ornith") ? 1u : 0u;
        return GUILD_STATUS_OK;
    }
    *yes = (has_integer(r,"qwen35moe.expert_count") || has_integer(r,"qwen2moe.expert_count")) ? 1u : 0u;
    return GUILD_STATUS_OK;
}
#if !defined(GUILD_BUNDLED_QWEN_RUNNERS)
guild_status_t recognize_qwen2moe(const guild_model_reader_v1_t* r, uint32_t* yes) {
    if (!r || r->struct_size<sizeof(*r) || !yes || !r->get_string) return GUILD_STATUS_ERR_ABI_MISMATCH;
    const auto arch=lower_text(text(r,"general.architecture",""));
    if (!arch.empty()) {
        *yes = (arch=="qwen2moe") ? 1u : 0u;
        return GUILD_STATUS_OK;
    }
    *yes = has_integer(r,"qwen2moe.expert_count") ? 1u : 0u;
    return GUILD_STATUS_OK;
}
guild_status_t recognize_ornith(const guild_model_reader_v1_t* r, uint32_t* yes) {
    if (!r || r->struct_size<sizeof(*r) || !yes || !r->get_string) return GUILD_STATUS_ERR_ABI_MISMATCH;
    const auto arch=lower_text(text(r,"general.architecture",""));
    *yes = (arch=="ornith") ? 1u : 0u;
    return GUILD_STATUS_OK;
}
#endif
guild_status_t recognize(const guild_model_reader_v1_t* r, uint32_t* yes) {
    uint32_t y4=0, y35=0;
    const auto s4 = recognize_qwen4exp(r, &y4);
    if (s4 != GUILD_STATUS_OK) return s4;
    const auto s35 = recognize_qwen35moe(r, &y35);
    if (s35 != GUILD_STATUS_OK) return s35;
    *yes = (y4 || y35) ? 1u : 0u;
    return GUILD_STATUS_OK;
}
guild_status_t fail(guild_error_info_t* e,const char* message) {
    if(e) { e->struct_size=sizeof(*e); e->code=GUILD_STATUS_ERR_INVALID_TENSOR;
        std::snprintf(e->message,sizeof(e->message),"%s",message); }
    return GUILD_STATUS_ERR_INVALID_TENSOR;
}
guild_status_t describe(const guild_model_reader_v1_t* r,guild_runner_model_v1_t* m,guild_error_info_t* e) {
    uint32_t yes=0;
    if(!m || m->struct_size<sizeof(*m) || recognize(r,&yes)!=GUILD_STATUS_OK || !yes ||
       !r->get_i64 || !r->get_f64 || !r->get_tensor) return GUILD_STATUS_ERR_ABI_MISMATCH;
    const auto arch=lower_text(text(r,"general.architecture",""));
    const bool indexed=(arch=="qwen4exp" || arch=="qwen3.8-flash-next" || (arch.empty() && has_integer(r,"qwen4exp.expert_count")));
    const std::string p=indexed ? "qwen4exp" : (arch=="qwen2moe" || has_integer(r,"qwen2moe.expert_count")) ? "qwen2moe" : "qwen35moe";
    struct ReqKey { const char* suffix; int64_t want; };
    const ReqKey reqs[] = {
        {".block_count", indexed ? 48 : 0},
        {".embedding_length", indexed ? 2560 : 0},
        {".expert_count", 0},
        {".expert_used_count", 0},
        {".attention.head_count", indexed ? 24 : 0},
        {".attention.head_count_kv", indexed ? 2 : 0},
    };
    for (const auto& req : reqs) {
        int64_t got = 0;
        const std::string key = p + req.suffix;
        if (!has_integer(r, key, &got)) {
            return fail(e, ("missing " + key).c_str());
        }
        if (req.want > 0 && got != req.want) {
            return fail(e, (key + " = " + std::to_string(got) + ", expected " + std::to_string(req.want)).c_str());
        }
    }
    *m={}; m->struct_size=sizeof(*m); m->version=GUILD_RUNNER_MODEL_VERSION_V1;
    std::snprintf(m->arch_id,sizeof(m->arch_id),"%s",p.c_str());
    std::snprintf(m->name,sizeof(m->name),"%s",text(r,"general.name",p.c_str()).c_str());
    m->n_embd=integer(r,p+".embedding_length",indexed?2560:2048);
    m->n_layers=integer(r,p+".block_count",indexed?48:40)-integer(r,p+".nextn_predict_layers",0);
    m->n_heads=integer(r,p+".attention.head_count",indexed?24:16);
    m->n_kv_heads=integer(r,p+".attention.head_count_kv",2);
    m->head_dim=integer(r,p+".attention.key_length",256);
    m->context_length=integer(r,p+".context_length",262144);
    m->full_attention_interval=integer(r,p+".full_attention_interval",4);
    m->n_experts=integer(r,p+".expert_count",indexed?512:256);
    m->active_experts=integer(r,p+".expert_used_count",indexed?10:8);
    m->expert_ff=integer(r,p+".expert_feed_forward_length",integer(r,p+".feed_forward_length",indexed?640:512));
    m->shared_ff=integer(r,p+".expert_shared_feed_forward_length",indexed?2560:512);
    m->shared_experts=(m->shared_ff>0 || has_integer(r,p+".expert_shared_count"))?1:0;
    m->ssm_state_size=integer(r,p+".ssm.state_size",128);
    m->ssm_k_heads=integer(r,p+".ssm.group_count",16);
    m->ssm_v_heads=integer(r,p+".ssm.time_step_rank",indexed?48:32);
    m->ssm_d_conv=integer(r,p+".ssm.conv_kernel",4);
    m->ssm_conv_channels=2*m->ssm_state_size*m->ssm_k_heads+m->ssm_state_size*m->ssm_v_heads;
    m->ssm_value_dim=m->ssm_state_size*m->ssm_v_heads;
    m->indexed_attention=indexed; m->gdn_gate_silu=!indexed;
    m->index_heads=indexed?integer(r,p+".attention.indexer.head_count",4):0;
    m->index_dim=indexed?integer(r,p+".attention.indexer.key_length",128):0;
    m->residual_streams=indexed?integer(r,p+".hyperconnection.count",4):1;
    m->residual_rank=indexed?integer(r,p+".hyperconnection.low_rank_dim",320):0;
    m->requires_ple=indexed; m->supports_mtp=1;
    m->prefill_strategy=indexed?GUILD_RUNNER_PREFILL_CHUNKED:GUILD_RUNNER_PREFILL_LAYER_BATCHED;
    m->rope_base=number(r,p+".rope.freq_base",indexed?1000000.0:10000000.0);
    m->rope_factor=number(r,p+".rope.scaling.factor",1.0);
    m->rope_original_context=number(r,p+".rope.scaling.original_context_length",0.0);
    const auto rope=text(r,(p+".rope.scaling.type").c_str(),"none");
    m->rope_kind=rope=="linear"?1:rope=="yarn"?2:0;
    std::strcpy(m->embedding_tensor,"token_embd.weight"); std::strcpy(m->output_tensor,"output.weight");
    std::strcpy(m->output_norm_tensor,"output_norm.weight"); std::strcpy(m->ple_table_tensor,"per_layer_token_embd.weight");
    std::strcpy(m->ple_key_tensor,"blk.1.ple_key.weight"); std::strcpy(m->layer_prefix,"blk.");
    guild_tensor_desc_t embedding{},head{};
    embedding.struct_size=sizeof(embedding); head.struct_size=sizeof(head);
    if(r->get_tensor(r->user,m->embedding_tensor,&embedding)!=GUILD_STATUS_OK ||
       r->get_tensor(r->user,m->output_tensor,&head)!=GUILD_STATUS_OK || embedding.n_dims!=2 || head.n_dims!=2 ||
       embedding.dims[0]!=m->n_embd || head.dims[0]!=m->n_embd || head.dims[1]!=embedding.dims[1])
        return fail(e,"Qwen runner: incompatible embedding or output matrix");
    m->vocab_size=embedding.dims[1];
    if(m->n_layers<=0 || m->full_attention_interval<=0 || m->n_layers%m->full_attention_interval ||
       m->n_experts<=0 || m->active_experts<=0 || m->active_experts>m->n_experts ||
       m->n_kv_heads<=0 || m->n_heads%m->n_kv_heads || m->head_dim!=256 ||
       m->ssm_state_size!=128 || m->ssm_d_conv!=4 ||
       (indexed && (m->residual_streams!=4 || m->residual_rank!=320 || m->index_heads!=4 || m->index_dim!=128)))
        return fail(e,"Qwen runner: unsupported hybrid topology or dimensions");
    if(indexed) {
        const bool any_ple_meta = has_integer(r,"qwen4exp.ple.vocab_per_layer") ||
                                  has_integer(r,"qwen4exp.ple.embedding_length") ||
                                  has_integer(r,"qwen4exp.ple.key_length") ||
                                  has_integer(r,"qwen4exp.ple.ngram_min") ||
                                  has_integer(r,"qwen4exp.ple.ngram_max");
        if (any_ple_meta) {
            int64_t pv=0, pe=0, pk=0, pmin=0, pmax=0;
            if (!has_integer(r,"qwen4exp.ple.vocab_per_layer",&pv) || pv != 20000096 ||
                !has_integer(r,"qwen4exp.ple.embedding_length",&pe) || pe != 2560 ||
                !has_integer(r,"qwen4exp.ple.key_length",&pk) || pk != 256 ||
                !has_integer(r,"qwen4exp.ple.ngram_min",&pmin) || pmin != 2 ||
                !has_integer(r,"qwen4exp.ple.ngram_max",&pmax) || pmax != 3) {
                return fail(e,"Qwen4Exp runner: invalid or incomplete PLE metadata");
            }
        }
        const bool any_artifact_ple_meta = has_integer(r,"qwen4exp.ple.ngram_size") ||
                                           has_integer(r,"qwen4exp.ple.heads_per_ngram") ||
                                           has_integer(r,"qwen4exp.ple.conv_kernel") ||
                                           has_integer(r,"qwen4exp.ple.eos_token_id");
        if (any_artifact_ple_meta) {
            int64_t ngs=0, hpn=0, ck=0, eos=0;
            if (!has_integer(r,"qwen4exp.ple.ngram_size",&ngs) || ngs != 3 ||
                !has_integer(r,"qwen4exp.ple.heads_per_ngram",&hpn) || hpn != 8 ||
                !has_integer(r,"qwen4exp.ple.conv_kernel",&ck) || ck != 4 ||
                !has_integer(r,"qwen4exp.ple.eos_token_id",&eos) || eos != 248044) {
                return fail(e,"Qwen4Exp runner: invalid or incomplete PLE artifact metadata");
            }
        }
        if (r->get_tensor) {
            guild_tensor_desc_t ple{}; ple.struct_size=sizeof(ple);
            const auto ple_st = r->get_tensor(r->user,m->ple_table_tensor,&ple);
            if(ple_st == GUILD_STATUS_OK) {
                if(ple.n_dims!=2 || ple.dims[0]!=160 || ple.dims[1]<=0)
                    return fail(e,"Qwen4Exp runner: invalid PLE tensor dimensions");
            }
        }
    }
    return GUILD_STATUS_OK;
}
guild_status_t finite(const float* v,size_t n,guild_error_info_t* e) {
    if(!v || n==0) {
        if(e) {
            e->struct_size=sizeof(*e);
            e->code=GUILD_STATUS_ERR_INVALID_ARGUMENT;
            std::snprintf(e->message,sizeof(e->message),"Qwen runner: missing output");
        }
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    static_assert(sizeof(float)==sizeof(uint32_t), "IEEE binary32 float required");
    size_t i=0;
    for(; n - i >= 64; i += 64) {
        uint32_t nonfinite = 0;
        for(size_t j=0; j<64; ++j) {
            uint32_t bits;
            std::memcpy(&bits, v + i + j, sizeof(bits));
            nonfinite |= (bits & 0x7f800000u) == 0x7f800000u;
        }
        if(nonfinite) {
            for(size_t j=0; j<64; ++j) {
                uint32_t bits;
                std::memcpy(&bits, v + i + j, sizeof(bits));
                if((bits & 0x7f800000u) == 0x7f800000u) {
                    if(e) {
                        e->struct_size=sizeof(*e);
                        e->code=GUILD_STATUS_ERR_NONFINITE_OUTPUT;
                        std::snprintf(e->message,sizeof(e->message),"Qwen runner: nonfinite value at element %zu",i+j);
                    }
                    return GUILD_STATUS_ERR_NONFINITE_OUTPUT;
                }
            }
        }
    }
    for(; i<n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, v + i, sizeof(bits));
        if((bits & 0x7f800000u) == 0x7f800000u) {
            if(e) {
                e->struct_size=sizeof(*e);
                e->code=GUILD_STATUS_ERR_NONFINITE_OUTPUT;
                std::snprintf(e->message,sizeof(e->message),"Qwen runner: nonfinite value at element %zu",i);
            }
            return GUILD_STATUS_ERR_NONFINITE_OUTPUT;
        }
    }
    return GUILD_STATUS_OK;
}
const guild_op_kind_t ops_qwen35moe[]={
    GUILD_OP_EMBEDDING_GATHER,GUILD_OP_QUANTIZE_ACT,GUILD_OP_QUANTIZED_GEMV,
    GUILD_OP_QUANTIZED_GEMM,GUILD_OP_MOE_ROUTER_TOPK,GUILD_OP_MOE_DISPATCH,
    GUILD_OP_BATCHED_EXPERT_EXEC,GUILD_OP_SHARED_EXPERT,GUILD_OP_MOE_COMBINE,
    GUILD_OP_RMS_NORM,GUILD_OP_ROPE_APPLY,GUILD_OP_KV_APPEND,GUILD_OP_ATTENTION_DECODE,
    GUILD_OP_GDN_CONV_L2,GUILD_OP_GDN_AB_PROJ,GUILD_OP_GDN_RECURRENCE_STEP,
    GUILD_OP_GDN_OUT_NORM,GUILD_OP_SPEC_VERIFY,GUILD_OP_FINITE_CHECK
};
const guild_op_kind_t ops_qwen4exp[]={
    GUILD_OP_EMBEDDING_GATHER,GUILD_OP_QUANTIZE_ACT,GUILD_OP_QUANTIZED_GEMV,
    GUILD_OP_QUANTIZED_GEMM,GUILD_OP_MOE_ROUTER_TOPK,GUILD_OP_MOE_DISPATCH,
    GUILD_OP_BATCHED_EXPERT_EXEC,GUILD_OP_SHARED_EXPERT,GUILD_OP_MOE_COMBINE,
    GUILD_OP_RMS_NORM,GUILD_OP_GATED_RESIDUAL,GUILD_OP_ROPE_APPLY,
    GUILD_OP_KV_APPEND,GUILD_OP_KV_GATHER,GUILD_OP_QSA_INDEXER_SELECT,
    GUILD_OP_ATTENTION_DECODE,GUILD_OP_GDN_CONV_L2,GUILD_OP_GDN_AB_PROJ,
    GUILD_OP_GDN_RECURRENCE_STEP,GUILD_OP_GDN_OUT_NORM,GUILD_OP_PLE_COMPUTE,
    GUILD_OP_SPEC_VERIFY,GUILD_OP_FINITE_CHECK
};
#if !defined(GUILD_BUNDLED_QWEN_RUNNERS)
const guild_runner_model_ext_v1_t ext_qwen35moe={sizeof(ext_qwen35moe),GUILD_RUNNER_MODEL_TAG_V1,1,250,recognize_qwen35moe,describe,finite};
const guild_runner_model_ext_v1_t ext_qwen4exp={sizeof(ext_qwen4exp),GUILD_RUNNER_MODEL_TAG_V1,1,250,recognize_qwen4exp,describe,finite};
const guild_runner_model_ext_v1_t ext_qwen2moe={sizeof(ext_qwen2moe),GUILD_RUNNER_MODEL_TAG_V1,1,250,recognize_qwen2moe,describe,finite};
const guild_runner_model_ext_v1_t ext_ornith={sizeof(ext_ornith),GUILD_RUNNER_MODEL_TAG_V1,1,250,recognize_ornith,describe,finite};
const guild_runner_model_ext_v1_t* const exts[4]={&ext_qwen35moe,&ext_qwen4exp,&ext_qwen2moe,&ext_ornith};
guild_status_t init(guild_core_host_t host,const guild_host_api_v1_t* api,guild_error_info_t* e) {
    if(!api || api->struct_size<sizeof(*api)) return GUILD_STATUS_ERR_ABI_MISMATCH;
    const char* architectures[]={"qwen35moe","qwen4exp","qwen2moe","ornith"};
    const char* ids[]={"runner.external.qwen35moe.v1","runner.external.qwen4exp.v1","runner.external.qwen2moe.v1","runner.external.ornith.v1"};
    for(size_t i=0;i<4;++i) {
        const bool is_qwen4 = (i==1);
        guild_runner_desc_v1_t d{};
        d.struct_size=sizeof(d); d.abi_version=GUILD_KERNEL_ABI_VERSION_V1;
        d.arch_id=architectures[i]; d.runner_id=ids[i]; d.package_id="org.guild.runners.qwen";
        d.runner_version=GUILD_KERNEL_ABI_MAKE_VERSION(1,1,0);
        d.required_core_caps=GUILD_CORE_CAP_CUDA_GRAPH_VERIFY |
            (is_qwen4 ? (GUILD_CORE_CAP_INDEXED_QSA_ATTN|GUILD_CORE_CAP_PLE_NGRAM)
                      : (GUILD_CORE_CAP_DENSE_CAUSAL_ATTN|GUILD_CORE_CAP_BATCHED_PREFILL));
        d.required_op_kinds=is_qwen4 ? ops_qwen4exp : ops_qwen35moe;
        d.n_required_op_kinds=is_qwen4 ? (sizeof(ops_qwen4exp)/sizeof(ops_qwen4exp[0]))
                                       : (sizeof(ops_qwen35moe)/sizeof(ops_qwen35moe[0]));
        d.next_ext=exts[i];
        const auto status=api->register_runner(host,&d,e);
        if(status!=GUILD_STATUS_OK) return status;
    }
    return GUILD_STATUS_OK;
}
#else
const guild_runner_model_ext_v1_t ext_bundled_qwen4exp={sizeof(ext_bundled_qwen4exp),GUILD_RUNNER_MODEL_TAG_V1,1,0,recognize_qwen4exp,describe,finite};
const guild_runner_model_ext_v1_t ext_bundled_qwen35moe={sizeof(ext_bundled_qwen35moe),GUILD_RUNNER_MODEL_TAG_V1,1,0,recognize_qwen35moe,describe,finite};
#endif
}
#if defined(GUILD_BUNDLED_QWEN_RUNNERS)
extern "C" guild_status_t guild_register_bundled_qwen_runners(guild_core_host_t host,const guild_host_api_v1_t* api,guild_error_info_t* e) {
    if(!api || api->struct_size<sizeof(*api) || !api->register_runner) return GUILD_STATUS_ERR_ABI_MISMATCH;
    {
        guild_runner_desc_v1_t d{};
        d.struct_size=sizeof(d); d.abi_version=GUILD_KERNEL_ABI_VERSION_V1;
        d.arch_id="qwen4exp"; d.runner_id="runner.qwen4exp.v1"; d.package_id="builtin.guild-kernels-qwen";
        d.runner_version=GUILD_KERNEL_ABI_VERSION_V1;
        d.required_core_caps=GUILD_CORE_CAP_INDEXED_QSA_ATTN|GUILD_CORE_CAP_CUDA_GRAPH_VERIFY|GUILD_CORE_CAP_PLE_NGRAM;
        d.required_op_kinds=ops_qwen4exp; d.n_required_op_kinds=sizeof(ops_qwen4exp)/sizeof(ops_qwen4exp[0]);
        d.next_ext=&ext_bundled_qwen4exp;
        const auto st=api->register_runner(host,&d,e);
        if(st!=GUILD_STATUS_OK) return st;
    }
    {
        guild_runner_desc_v1_t d{};
        d.struct_size=sizeof(d); d.abi_version=GUILD_KERNEL_ABI_VERSION_V1;
        d.arch_id="qwen35moe"; d.runner_id="runner.qwen35moe.v1"; d.package_id="builtin.guild-kernels-ornith";
        d.runner_version=GUILD_KERNEL_ABI_VERSION_V1;
        d.required_core_caps=GUILD_CORE_CAP_DENSE_CAUSAL_ATTN|GUILD_CORE_CAP_CUDA_GRAPH_VERIFY|GUILD_CORE_CAP_BATCHED_PREFILL;
        d.required_op_kinds=ops_qwen35moe; d.n_required_op_kinds=sizeof(ops_qwen35moe)/sizeof(ops_qwen35moe[0]);
        d.next_ext=&ext_bundled_qwen35moe;
        const auto st=api->register_runner(host,&d,e);
        if(st!=GUILD_STATUS_OK) return st;
    }
    return GUILD_STATUS_OK;
}
#else
extern "C" GUILD_PLUGIN_EXPORT guild_status_t guild_plugin_entry_v1(uint32_t abi,guild_plugin_descriptor_v1_t* d,guild_error_info_t*) {
    if(GUILD_ABI_VERSION_MAJOR(abi)!=1 || !d) return GUILD_STATUS_ERR_ABI_MISMATCH;
    *d={};d->struct_size=sizeof(*d);d->abi_version=GUILD_KERNEL_ABI_VERSION_V1;
    d->package_id="org.guild.runners.qwen";d->package_version=GUILD_PACKAGE_VERSION;
    d->publisher_id="org.guild-inference";d->init=init;
    return GUILD_STATUS_OK;
}
#endif
