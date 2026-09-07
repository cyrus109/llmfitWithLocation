//! Recommended LM Studio "Load" settings for a model on this machine.
//!
//! LM Studio drives llama.cpp (GGUF) or MLX, so every knob maps to a
//! llama.cpp parameter. The numbers come from the same memory model the fit
//! table uses (`LlmModel::kv_cache_gb`, `quant_bpp`), so the detail panel and
//! the ranking never disagree about what fits.

use crate::fit::{ModelFit, RunMode};
use crate::hardware::{GpuBackend, SystemSpecs};
use crate::models::{KvQuant, quant_bpp};

/// One recommended setting: LM Studio's label, the value to enter, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Setting {
    pub label: &'static str,
    pub value: String,
    pub why: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LmStudioSettings {
    pub settings: Vec<Setting>,
}

/// Round a token budget down to the multiples LM Studio's slider snaps to.
fn snap_context(tokens: u32) -> u32 {
    if tokens >= 8192 {
        tokens / 4096 * 4096
    } else if tokens >= 2048 {
        tokens / 1024 * 1024
    } else {
        tokens.max(512)
    }
}

/// Tokens of KV cache that fit in `budget_gb` for this model at `kv`.
fn context_for_budget(fit: &ModelFit, budget_gb: f64, kv: KvQuant) -> u32 {
    const REF_CTX: u32 = 4096;
    let m = &fit.model;
    let per_token =
        (m.kv_cache_gb(REF_CTX, kv) - m.kv_cache_gb(0, kv)).max(0.0) / f64::from(REF_CTX);
    if per_token <= 0.0 {
        return m.context_length;
    }
    ((budget_gb.max(0.0) / per_token) as u32).min(m.context_length)
}

pub fn recommend(fit: &ModelFit, specs: &SystemSpecs) -> LmStudioSettings {
    let m = &fit.model;
    let weights_gb = m.params_b() * quant_bpp(&fit.best_quant);
    let overhead_gb = 0.5;
    let gpu_backend = matches!(
        specs.backend,
        GpuBackend::Cuda
            | GpuBackend::Metal
            | GpuBackend::Rocm
            | GpuBackend::Vulkan
            | GpuBackend::Sycl
    );
    let on_gpu = specs.has_gpu && gpu_backend && !matches!(fit.run_mode, RunMode::CpuOnly);

    // GPU pool: on unified memory the Metal working-set cap, else VRAM.
    let gpu_pool = if specs.unified_memory {
        specs.gpu_available_gb.or(specs.gpu_vram_gb)
    } else {
        specs.total_gpu_vram_gb.or(specs.gpu_vram_gb)
    }
    .unwrap_or(0.0);

    // ── GPU offload (layers) ──────────────────────────────────────────
    let n_layers = m.num_hidden_layers;
    let (offload_layers, offload_all) = if !on_gpu {
        (0u32, false)
    } else {
        match fit.run_mode {
            RunMode::Gpu | RunMode::MoeOffload | RunMode::TensorParallel => {
                (n_layers.unwrap_or(0), true)
            }
            _ => {
                // Partial: layers whose weights fit next to a 4k KV window.
                let kv4k = m.kv_cache_gb(4096, KvQuant::Fp16);
                let room = (gpu_pool - kv4k - overhead_gb).max(0.0);
                let frac = (room / weights_gb).clamp(0.0, 1.0);
                match n_layers {
                    Some(n) => ((f64::from(n) * frac).floor() as u32, frac >= 1.0),
                    None => (0, false),
                }
            }
        }
    };

    // ── Context length ────────────────────────────────────────────────
    // Leave 10% of the pool free: LM Studio's compute buffers grow with
    // n_batch and the fit estimate does not include them.
    let pool = fit.memory_available_gb;
    let weights_in_pool = if on_gpu && !offload_all && !specs.unified_memory {
        // CPU-offload path: the pool is RAM, holding the non-offloaded share.
        weights_gb * (1.0 - offload_layers as f64 / n_layers.unwrap_or(1).max(1) as f64)
    } else {
        weights_gb
    };
    let kv_budget = pool * 0.9 - weights_in_pool - overhead_gb;
    let ctx_fp16 = snap_context(context_for_budget(fit, kv_budget, KvQuant::Fp16));
    let ctx_q8 = snap_context(context_for_budget(fit, kv_budget, KvQuant::Q8_0));
    let native = m.context_length;

    // ── KV cache quant ────────────────────────────────────────────────
    // Only worth it when fp16 KV is what caps the window, and only where
    // llama.cpp's quantised KV works (needs flash attention on a GPU).
    let want_kv_quant =
        gpu_backend && on_gpu && ctx_fp16 < native && ctx_fp16 < 32768 && ctx_q8 > ctx_fp16;
    let context = if want_kv_quant { ctx_q8 } else { ctx_fp16 };

    // ── Threads ───────────────────────────────────────────────────────
    let cores = specs.total_cpu_cores.max(1);
    // ponytail: 75% of logical cores. Hyperthreads and efficiency cores slow
    // llama.cpp's compute threads down; a P-core count would be better.
    let threads = if offload_all {
        cores.clamp(1, 8)
    } else {
        (cores * 3 / 4).max(1)
    };

    // ── Batch sizes ───────────────────────────────────────────────────
    let (n_batch, n_ubatch) = if offload_all && gpu_pool >= 12.0 {
        (2048, 512)
    } else if on_gpu {
        (1024, 512)
    } else {
        (512, 256)
    };

    // ── mlock ─────────────────────────────────────────────────────────
    let keep_in_memory = weights_gb <= specs.total_ram_gb * 0.6;

    let kv_on_gpu = offload_all
        || (on_gpu
            && gpu_pool
                - weights_gb * offload_layers as f64 / n_layers.unwrap_or(1).max(1) as f64
                - overhead_gb
                >= m.kv_cache_gb(context, KvQuant::Fp16));

    let yes_no = |b: bool| if b { "On" } else { "Off" }.to_string();
    let mut s = Vec::new();

    s.push(Setting {
        label: "Context Length",
        value: format!("{context}"),
        why: if context >= native {
            format!("full native window fits ({} GB pool)", fmt_gb(pool))
        } else if want_kv_quant {
            format!("with q8_0 KV; fp16 KV caps at {ctx_fp16} (native {native})")
        } else {
            format!(
                "KV cache budget {} GB after weights (native {native})",
                fmt_gb(kv_budget.max(0.0))
            )
        },
    });
    s.push(Setting {
        label: "GPU Offload",
        value: match (n_layers, on_gpu) {
            (_, false) => "0".to_string(),
            (Some(n), true) if offload_all => format!("{n} (max)"),
            (Some(_), true) => format!("{offload_layers}"),
            (None, true) => "max".to_string(),
        },
        why: if !on_gpu {
            "no usable GPU for this model; runs on CPU".to_string()
        } else if offload_all {
            format!(
                "{} weights fit in {} GB GPU memory",
                fit.best_quant,
                fmt_gb(gpu_pool)
            )
        } else {
            format!(
                "{} GB weights vs {} GB GPU memory; rest in RAM",
                fmt_gb(weights_gb),
                fmt_gb(gpu_pool)
            )
        },
    });
    s.push(Setting {
        label: "CPU Thread Pool Size",
        value: format!("{threads}"),
        why: if offload_all {
            format!("GPU does the work; {cores} cores detected")
        } else {
            format!("~75% of {cores} cores, leaves room for the OS")
        },
    });
    s.push(Setting {
        label: "Evaluation Batch Size",
        value: format!("{n_batch}"),
        why: "prompt-processing chunk; larger = faster prefill, more VRAM".into(),
    });
    s.push(Setting {
        label: "Physical Batch Size",
        value: format!("{n_ubatch}"),
        why: "llama.cpp default; raise only with plenty of VRAM".into(),
    });
    s.push(Setting {
        label: "Max Concurrent Predictions",
        value: "1".into(),
        why: "raise only when several clients share the server".into(),
    });
    s.push(Setting {
        label: "Unified KV Cache",
        value: "Off".into(),
        why: "only useful with concurrent predictions > 1".into(),
    });
    s.push(Setting {
        label: "RoPE Frequency Base / Scale",
        value: "Auto".into(),
        why: format!("read from the GGUF; change only to stretch past {native}"),
    });
    s.push(Setting {
        label: "Offload KV Cache to GPU Memory",
        value: yes_no(kv_on_gpu),
        why: if kv_on_gpu {
            "KV fits beside the weights on the GPU".into()
        } else {
            "GPU memory is full; keep KV in RAM".into()
        },
    });
    s.push(Setting {
        label: "Keep Model in Memory",
        value: yes_no(keep_in_memory),
        why: format!(
            "{} GB weights vs {} GB RAM (mlock pins pages)",
            fmt_gb(weights_gb),
            fmt_gb(specs.total_ram_gb)
        ),
    });
    s.push(Setting {
        label: "Try mmap()",
        value: "On".into(),
        why: "faster load, pages in from disk on demand".into(),
    });
    s.push(Setting {
        label: "Speculative Decoding",
        value: "Off".into(),
        why: "needs a matching small draft model; measure before enabling".into(),
    });
    s.push(Setting {
        label: "Flash Attention",
        value: yes_no(on_gpu),
        why: if on_gpu {
            "less memory per token, required for KV quantisation".into()
        } else {
            "no gain on CPU".into()
        },
    });
    s.push(Setting {
        label: "K / V Cache Quantization",
        value: if want_kv_quant {
            "q8_0 / q8_0".into()
        } else {
            "Off (fp16)".into()
        },
        why: if want_kv_quant {
            format!("halves KV: {ctx_fp16} → {ctx_q8} tokens at ~no quality cost")
        } else if ctx_fp16 >= native {
            "fp16 KV already fits the full window".into()
        } else {
            "fp16 fits enough context; quantised KV not needed".into()
        },
    });

    LmStudioSettings { settings: s }
}

fn fmt_gb(gb: f64) -> String {
    format!("{gb:.1}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::LlmModel;

    fn specs(ram: f64, vram: Option<f64>, unified: bool, backend: GpuBackend) -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: ram,
            available_ram_gb: ram * 0.8,
            total_cpu_cores: 16,
            cpu_name: "Test CPU".to_string(),
            has_gpu: vram.is_some(),
            gpu_vram_gb: vram,
            total_gpu_vram_gb: vram,
            gpu_available_gb: vram.map(|v| v * 0.75),
            gpu_name: vram.map(|_| "Test GPU".to_string()),
            gpu_count: u32::from(vram.is_some()),
            unified_memory: unified,
            backend,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    fn model() -> LlmModel {
        LlmModel {
            name: "Qwen/Qwen2.5-7B-Instruct".to_string(),
            provider: "Alibaba".to_string(),
            parameter_count: "7B".to_string(),
            parameters_raw: Some(7_600_000_000),
            min_ram_gb: 8.0,
            recommended_ram_gb: 16.0,
            min_vram_gb: Some(8.0),
            quantization: "Q4_K_M".to_string(),
            context_length: 131072,
            use_case: "Chat".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: vec![],
            capabilities: vec![],
            languages: vec![],
            format: crate::models::ModelFormat::default(),
            num_attention_heads: Some(28),
            num_key_value_heads: Some(4),
            num_hidden_layers: Some(28),
            head_dim: Some(128),
            attention_layout: None,
            hidden_size: None,
            moe_intermediate_size: None,
            vocab_size: None,
            shared_expert_intermediate_size: None,
            architecture: None,
            license: None,
        }
    }

    fn get<'a>(r: &'a LmStudioSettings, label: &str) -> &'a str {
        &r.settings.iter().find(|s| s.label == label).unwrap().value
    }

    #[test]
    fn cuda_gpu_offloads_everything_and_caps_context_by_vram() {
        let sp = specs(64.0, Some(24.0), false, GpuBackend::Cuda);
        let fit = ModelFit::analyze(&model(), &sp);
        let r = recommend(&fit, &sp);
        assert_eq!(get(&r, "GPU Offload"), "28 (max)");
        assert_eq!(get(&r, "Flash Attention"), "On");
        let ctx: u32 = get(&r, "Context Length").parse().unwrap();
        assert!(
            (8192..=131072).contains(&ctx) && ctx.is_multiple_of(4096),
            "ctx={ctx}"
        );
    }

    #[test]
    fn cpu_only_box_gets_zero_offload_and_no_flash_attention() {
        let sp = specs(32.0, None, false, GpuBackend::CpuX86);
        let fit = ModelFit::analyze(&model(), &sp);
        let r = recommend(&fit, &sp);
        assert_eq!(get(&r, "GPU Offload"), "0");
        assert_eq!(get(&r, "Flash Attention"), "Off");
        assert_eq!(get(&r, "CPU Thread Pool Size"), "12");
        assert_eq!(get(&r, "Offload KV Cache to GPU Memory"), "Off");
    }

    #[test]
    fn snap_rounds_down_to_slider_steps() {
        assert_eq!(snap_context(131072), 131072);
        assert_eq!(snap_context(9000), 8192);
        assert_eq!(snap_context(3000), 2048);
        assert_eq!(snap_context(100), 512);
    }
}
