#include <guild/abi/runner_abi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" guild_status_t guild_plugin_entry_v1(uint32_t,guild_plugin_descriptor_v1_t*,guild_error_info_t*);
#define CHECK(x) do { if(!(x)) { std::fprintf(stderr,"failed %s:%d: %s\n",__FILE__,__LINE__,#x); std::abort(); } } while(0)
static int count=0;
static guild_status_t collect(guild_core_host_t,const guild_runner_desc_v1_t* d,guild_error_info_t*) {
    CHECK(d->next_ext && d->package_id && d->n_required_op_kinds);
    const auto* ext=static_cast<const guild_runner_model_ext_v1_t*>(d->next_ext);
    CHECK(ext->struct_size==sizeof(*ext) && ext->priority>0);
    guild_model_reader_v1_t reader{};reader.struct_size=sizeof(reader);
    reader.get_string=[](void*,const char* k,char* out,size_t n)->guild_status_t {
        if(std::strcmp(k,"general.architecture")) return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        std::snprintf(out,n,"qwen35moe");return GUILD_STATUS_OK;
    };
    reader.get_i64=[](void*,const char*,int64_t*)->guild_status_t{return GUILD_STATUS_ERR_INVALID_ARGUMENT;};
    reader.get_f64=[](void*,const char*,double*)->guild_status_t{return GUILD_STATUS_ERR_INVALID_ARGUMENT;};
    reader.get_tensor=[](void*,const char*,guild_tensor_desc_t* t)->guild_status_t {
        t->n_dims=2;t->dims[0]=2048;t->dims[1]=248320;return GUILD_STATUS_OK;
    };
    guild_runner_model_v1_t m{};m.struct_size=sizeof(m);
    CHECK(ext->describe(&reader,&m,nullptr)==GUILD_STATUS_OK);
    CHECK(m.residual_streams==1 && m.n_layers==40 && m.ssm_v_heads==32 && m.active_experts==8);
    CHECK(m.prefill_strategy==GUILD_RUNNER_PREFILL_LAYER_BATCHED && !m.requires_ple);
    m.struct_size=4;CHECK(ext->describe(&reader,&m,nullptr)==GUILD_STATUS_ERR_ABI_MISMATCH);
    ++count;return GUILD_STATUS_OK;
}
int main() {
    guild_plugin_descriptor_v1_t d{};
    CHECK(guild_plugin_entry_v1(GUILD_KERNEL_ABI_VERSION_V1,&d,nullptr)==GUILD_STATUS_OK);
    guild_host_api_v1_t host{};host.struct_size=sizeof(host);host.register_runner=collect;
    CHECK(d.init(nullptr,&host,nullptr)==GUILD_STATUS_OK && count==4);
    CHECK(guild_plugin_entry_v1(GUILD_MAKE_ABI_VERSION(2,0,0),&d,nullptr)==GUILD_STATUS_ERR_ABI_MISMATCH);
}
