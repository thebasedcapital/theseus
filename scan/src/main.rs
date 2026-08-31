//! theseus-scan — static artifact scanner for safetensors / GGUF / PEFT-LoRA (owner: ScanFormats).
//!
//! Emits Inspector-schema-v1-compatible JSON (same families/total/skipped/verdicts/preflight
//! shape as inspect/src/main.rs) plus `input_format`, `convention`, and
//! `quantization: {scheme, stats_source}` so a scan of a quantized file can never be confused
//! with a scan of an f16 one. Preflight reproduces inspect/'s operation matrix with the *same*
//! provisional threshold constants (transcribed below, marked PROVISIONAL).

mod formats;
mod gguf;
mod stats;

use std::collections::BTreeMap;
use std::env;
use std::process::exit;
use std::time::Instant;

use formats::{ArtifactKind, Container, Sniff};
use stats::StatAcc;

const CONVENTION: &str = "mean_of_per_tensor_ratios";

/// PROVISIONAL risk thresholds, transcribed from inspect/src/main.rs (lines ~32-39) so preflight
/// reproduces inspect/'s matrix byte-for-byte. Calibrated on M1's n=2 measured pairs and nowhere
/// else; `theseus` must re-fit them on the M3 history ledger (ROADMAP A6).
const T_QUANT_ABS: f64 = 0.0165; // 1.5x the 0.011 measured for dense Qwen2.5 families
const T_QUANT_TOTAL_ABS: f64 = 0.0168; // 1.5 x measured base total J (0.01123)
const T_EXPORT_FRAC: f64 = 0.02; // share of a family below f16 normal range
const T_ADAPT_DYN: f64 = 12.0; // log10 dynamic range across a family
const T_ADAPT_ROW: f64 = 2.0e5; // row-energy imbalance

#[derive(PartialEq)]
enum Mode {
    Inspect,
    Preflight,
}

fn usage() -> ! {
    eprintln!("usage: theseus-scan [inspect|preflight] <artifact> [--json OUT]");
    eprintln!();
    eprintln!("  reads a safetensors / GGUF / PEFT-LoRA artifact and prints per-family static");
    eprintln!("  conditioning (q4_block_mse, dyn_range_log10, row_energy_imbalance,");
    eprintln!("  frac_below_f16_normal, amax_over_rms) plus quantization provenance.");
    eprintln!("  preflight: operation x risk matrix (thresholds provisional, as inspect/).");
    exit(2)
}

fn js_esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn fmt_f64(v: f64) -> String {
    format!("{:.8}", v)
}

fn report_map(r: &BTreeMap<&'static str, f64>) -> String {
    let keys = [
        "q4_block_mse",
        "q4_block_mse_pooled",
        "dyn_range_log10",
        "row_energy_imbalance",
        "amax_over_rms",
        "below_f16_normal",
        "frac_below_f16_normal",
        "weights",
    ];
    keys.iter()
        .enumerate()
        .map(|(i, k)| {
            format!(
                "{}\"{}\": {}",
                if i > 0 { ", " } else { "" },
                k,
                fmt_f64(r[*k])
            )
        })
        .collect()
}

/// total's row_energy_imbalance is the worst-family value (pooling across tensors inflates it and
/// says nothing about conditioning; same override as inspect/src/main.rs).
fn total_with_override(
    per_family: &BTreeMap<&'static str, StatAcc>,
    total: &StatAcc,
) -> BTreeMap<&'static str, f64> {
    let mut t = total.report();
    let worst = per_family
        .values()
        .map(|a| a.report()["row_energy_imbalance"])
        .fold(0.0f64, f64::max);
    t.insert("row_energy_imbalance", worst);
    t
}

/// Provisional operation x risk matrix — transcription of inspect/src/main.rs::ops_matrix.
fn ops_matrix(
    per_family: &BTreeMap<&'static str, StatAcc>,
    total: &BTreeMap<&'static str, f64>,
) -> Vec<(String, &'static str, String)> {
    let mut out: Vec<(String, &'static str, String)> = Vec::new();
    let worst_q = per_family
        .iter()
        .map(|(f, a)| (*f, a.report()["q4_block_mse"]))
        .fold(("none", 0.0f64), |x, acc| if acc.1 > x.1 { acc } else { x });
    let worst_export = per_family
        .iter()
        .map(|(f, a)| (*f, a.report()["frac_below_f16_normal"]))
        .fold(("none", 0.0f64), |x, acc| if acc.1 > x.1 { acc } else { x });
    let worst_adapt = per_family
        .iter()
        .map(|(f, a)| (*f, a.report()["dyn_range_log10"]))
        .fold(("none", 0.0f64), |x, acc| if acc.1 > x.1 { acc } else { x });

    out.push((
        "export.gguf.f16".into(),
        if worst_export.1 > T_EXPORT_FRAC { "AT_RISK" } else { "OK" },
        format!(
            "{:.2}% of {} weights below the f16 normal range (limit {:.1}%)",
            100.0 * worst_export.1,
            worst_export.0,
            100.0 * T_EXPORT_FRAC
        ),
    ));
    for op in ["quantize.gguf.q8_0", "quantize.gguf.q5_k_m", "quantize.gguf.q4_k_m"] {
        out.push((
            op.into(),
            if worst_q.1 > T_QUANT_ABS || total["q4_block_mse"] > T_QUANT_TOTAL_ABS {
                "AT_RISK"
            } else {
                "OK"
            },
            format!(
                "worst family {} 4-bit block proxy {:.5} (limit {:.4}); total {:.5}",
                worst_q.0, worst_q.1, T_QUANT_ABS, total["q4_block_mse"]
            ),
        ));
    }
    out.push((
        "adapt.lora.r16".into(),
        if worst_adapt.1 > T_ADAPT_DYN { "AT_RISK" } else { "OK" },
        format!(
            "worst family {} dynamic range 1e{:.2} (limit 1e{:.1}); row-energy imbalance {:.3e}",
            worst_adapt.0,
            worst_adapt.1,
            T_ADAPT_DYN,
            per_family
                .values()
                .map(|a| a.report()["row_energy_imbalance"])
                .fold(0.0f64, f64::max)
        ),
    ));
    out.push((
        "merge.linear".into(),
        "UNAVAILABLE".into(),
        "needs a second checkpoint to compare coordinates against".into(),
    ));
    out.push((
        "merge.ties".into(),
        "UNAVAILABLE".into(),
        "needs task vectors against a shared base".into(),
    ));
    out.push((
        "quantize.awlora".into(),
        "UNAVAILABLE".into(),
        "static features do not predict adapter-on-quantized-base behaviour yet".into(),
    ));
    out
}

fn verdicts(per_family: &BTreeMap<&'static str, StatAcc>, tr: &BTreeMap<&'static str, f64>) -> Vec<String> {
    let mut flags: Vec<String> = Vec::new();
    for (fam, acc) in per_family {
        let r = acc.report();
        if r["q4_block_mse"] > T_QUANT_ABS {
            flags.push(format!(
                "QUANT_RISK {}: 4-bit block MSE proxy {:.5} > {:.4}",
                fam, r["q4_block_mse"], T_QUANT_ABS
            ));
        }
        if r["frac_below_f16_normal"] > T_EXPORT_FRAC {
            flags.push(format!(
                "F16_EXPORT_RISK {}: {:.3}% of weights below the f16 normal range",
                fam,
                100.0 * r["frac_below_f16_normal"]
            ));
        }
        if r["dyn_range_log10"] > T_ADAPT_DYN {
            flags.push(format!(
                "ADAPT_RISK {}: dynamic range 1e{:.1} > 1e{:.1}",
                fam, r["dyn_range_log10"], T_ADAPT_DYN
            ));
        }
        if r["row_energy_imbalance"] > T_ADAPT_ROW {
            flags.push(format!(
                "ADAPT_RISK {}: row-energy imbalance {:.3e} > {:.1e}",
                fam, r["row_energy_imbalance"], T_ADAPT_ROW
            ));
        }
    }
    if tr["q4_block_mse"] > T_QUANT_TOTAL_ABS {
        flags.push(format!(
            "QUANT_RISK TOTAL: 4-bit proxy {:.5} > {:.4} (1.5x pristine-family reference)",
            tr["q4_block_mse"], T_QUANT_TOTAL_ABS
        ));
    }
    flags
}

/// Derive the quantization scheme descriptor from the metered type counts. A single quant type
/// gets its canonical vendor name (q4_k -> "q4_k_m"); a mix is reported honestly as mixed:a+b.
fn scheme_from_counts(counts: &BTreeMap<&'static str, usize>) -> Option<String> {
    let quants: Vec<&str> = counts
        .iter()
        .filter(|(k, _)| **k != "f32" && **k != "f16" && **k != "bf16")
        .filter(|(_, c)| **c > 0)
        .map(|(k, _)| *k)
        .collect();
    if quants.is_empty() {
        return None;
    }
    if quants.len() == 1 {
        let canon = match quants[0] {
            "q4_k" => "q4_k_m",
            "q5_k" => "q5_k_m",
            "q6_k" => "q6_k",
            other => other,
        };
        Some(canon.to_string())
    } else {
        Some(format!("mixed:{}", quants.join("+")))
    }
}

fn render_table(per_family: &BTreeMap<&'static str, StatAcc>, total: &BTreeMap<&'static str, f64>) {
    println!(
        "{:<12} {:>9} {:>10} {:>12} {:>10} {:>11} {:>12}",
        "family", "q4_mse", "dyn_rng", "row_imbal", "amax/rms", "frac<f16n", "weights"
    );
    for (fam, acc) in per_family {
        let r = acc.report();
        println!(
            "{:<12} {:>9.5} {:>10.3} {:>12.1} {:>10.1} {:>11.6} {:>12}",
            fam,
            r["q4_block_mse"],
            r["dyn_range_log10"],
            r["row_energy_imbalance"],
            r["amax_over_rms"],
            r["frac_below_f16_normal"],
            r["weights"] as u64
        );
    }
    println!(
        "{:<12} {:>9.5} {:>10.3} {:>12.1} {:>10.1} {:>11.6} {:>12}",
        "TOTAL",
        total["q4_block_mse"],
        total["dyn_range_log10"],
        total["row_energy_imbalance"],
        total["amax_over_rms"],
        total["frac_below_f16_normal"],
        total["weights"] as u64
    );
    println!();
}

fn build_census_json(
    path: &str,
    per_family: &BTreeMap<&'static str, StatAcc>,
    total: &BTreeMap<&'static str, f64>,
    skipped: &[(String, String)],
    counts: &BTreeMap<&'static str, usize>,
    ifmt: &str,
    preflight: Option<&[(String, &'static str, String)]>,
) -> String {
    let scheme = scheme_from_counts(counts);
    let scheme_s = scheme
        .as_deref()
        .map(|s| format!("\"{}\"", s))
        .unwrap_or_else(|| "null".into());
    let stats_source = if scheme.is_some() { "structural" } else { "dequantized" };
    let mut j = format!(
        "{{\n  \"path\": \"{}\",\n  \"input_format\": \"{}\",\n  \"convention\": \"{}\",\n  \"families\": {{\n",
        js_esc(path),
        ifmt,
        CONVENTION
    );
    let mut first = true;
    for (fam, acc) in per_family {
        if !first {
            j.push_str(",\n");
        }
        first = false;
        j.push_str(&format!("    \"{}\": {{ {} }}", fam, report_map(&acc.report())));
    }
    j.push_str(&format!("\n  }},\n  \"total\": {{ {} }}", report_map(total)));
    j.push_str(&format!(
        ",\n  \"quantization\": {{ \"scheme\": {}, \"stats_source\": \"{}\" }}",
        scheme_s, stats_source
    ));
    j.push_str(",\n  \"quant_types\": {");
    let mut first = true;
    for (k, v) in counts {
        if !first {
            j.push_str(", ");
        }
        first = false;
        j.push_str(&format!("\"{}\": {}", k, v));
    }
    j.push_str("},\n  \"skipped\": [");
    for (i, (n, why)) in skipped.iter().take(12).enumerate() {
        if i > 0 {
            j.push_str(", ");
        }
        j.push_str(&format!("[\"{}\", \"{}\"]", js_esc(n), js_esc(why)));
    }
    j.push_str("],\n  \"verdicts\": [");
    let vds = verdicts(per_family, total);
    for (i, v) in vds.iter().enumerate() {
        if i > 0 {
            j.push_str(", ");
        }
        j.push_str(&format!("\"{}\"", js_esc(v)));
    }
    if let Some(ops) = preflight {
        j.push_str("],\n  \"preflight\": [\n    ");
        j.push_str(
            &ops.iter()
                .map(|(o, v, r)| format!("[\"{}\", \"{}\", \"{}\"]", o, v, js_esc(r)))
                .collect::<Vec<_>>()
                .join(",\n    "),
        );
        j.push_str("\n  ]\n}\n");
    } else {
        j.push_str("]\n}\n");
    }
    j
}

fn write_out(json: &str, json_out: Option<&str>) {
    if let Some(o) = json_out {
        if let Err(e) = std::fs::write(o, json) {
            eprintln!("error: writing {}: {}", o, e);
            exit(1);
        }
        eprintln!("wrote {}", o);
    }
}

fn adapter_json(path: &str, hdr: &formats::SafetensorsHeader) -> String {
    let a = &hdr.adapter;
    let mut j = String::from("{\n  \"path\": \"");
    j.push_str(&js_esc(path));
    j.push_str("\",\n  \"input_format\": \"adapter\",\n  \"kind\": \"peft_lora\",\n  \"families\": {},\n  \"total\": {},\n  \"adapter\": {");
    j.push_str(&format!(
        "\"rank\": {}, \"ranks\": {{ {} }}, \"target_modules\": [",
        a.rank.map(|r| r.to_string()).unwrap_or_else(|| "null".into()),
        a.ranks
            .iter()
            .map(|(r, c)| format!("\"{}\": {}", r, c))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    for (i, m) in a.target_modules.iter().enumerate() {
        if i > 0 {
            j.push_str(", ");
        }
        j.push_str(&format!("\"{}\"", js_esc(m)));
    }
    j.push_str(&format!(
        "], \"tensor_count\": {}, \"pair_count\": {}, \"dtype\": {}",
        a.tensor_count,
        a.pair_count,
        a.dtype.clone().map(|d| format!("\"{}\"", d)).unwrap_or_else(|| "null".into())
    ));
    j.push_str(", \"alpha\": null, \"alpha_reason\": \"");
    j.push_str(a.alpha_reason.unwrap_or("lora_alpha not present in safetensors bytes"));
    j.push_str("\"}");
    j.push_str(",\n  \"quantization\": { \"scheme\": null, \"stats_source\": null, \"reason\": \"adapters carry deltas, not weight blocks; census does not apply\" }");
    j.push_str(",\n  \"skipped\": [],\n  \"verdicts\": [\"");
    j.push_str(formats::ADAPTER_VERDICT);
    j.push_str("\"]\n}\n");
    j
}

fn unsupported_json(path: &str, sniff: &Sniff) -> String {
    let fmt = match sniff.container {
        Container::PyTorchBin => "pytorch_bin",
        Container::NumpyPair => "mlx_npy_npz",
        _ => "unknown",
    };
    format!(
        "{{\n  \"path\": \"{}\",\n  \"input_format\": \"{}\",\n  \"convention\": null,\n  \"families\": {{}},\n  \"total\": {{}},\n  \"quantization\": {{ \"scheme\": null, \"stats_source\": \"unavailable\", \"reason\": \"{}\" }},\n  \"skipped\": [[\"{}\", \"{}\"]],\n  \"verdicts\": [\"UNSUPPORTED: {}\"]\n}}\n",
        js_esc(path),
        fmt,
        js_esc(&sniff.reason),
        js_esc(path),
        js_esc(&sniff.reason),
        js_esc(&sniff.reason)
    )
}

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = env::args().collect();
    let mut mode = Mode::Inspect;
    let mut path: Option<String> = None;
    let mut json_out: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => {
                if let Some(v) = args.get(i + 1) {
                    json_out = Some(v.clone());
                }
                i += 2;
            }
            "-h" | "--help" => usage(),
            "inspect" | "preflight" => {
                mode = if args[i] == "preflight" { Mode::Preflight } else { Mode::Inspect };
                i += 1;
            }
            other => {
                path = Some(other.to_string());
                i += 1;
            }
        }
    }
    let path = match path {
        Some(p) => p,
        None => usage(),
    };

    let sniff = formats::sniff(&path);

    match (sniff.kind, sniff.container) {
        (ArtifactKind::Unsupported, _) => {
            let json = unsupported_json(&path, &sniff);
            println!("theseus-scan  {}  unsupported artifact", path);
            println!("  {}", sniff.reason);
            write_out(&json, json_out.as_deref());
        }
        (ArtifactKind::Adapter, Container::Safetensors) => {
            let hdr = match formats::parse_header(&path) {
                Ok(h) => h,
                Err(e) => fatal(&e),
            };
            let json = adapter_json(&path, &hdr);
            println!("theseus-scan  {}  PEFT/LoRA adapter", path);
            let a = &hdr.adapter;
            println!(
                "  lora_A={} lora_B={} rank={} targets={} tensors={} dtype={}",
                a.has_lora_a,
                a.has_lora_b,
                a.rank.map(|r| r.to_string()).unwrap_or_else(|| "?".into()),
                a.target_modules.join(","),
                a.tensor_count,
                a.dtype.clone().unwrap_or_else(|| "-".into())
            );
            println!("  verdict: {}", formats::ADAPTER_VERDICT);
            write_out(&json, json_out.as_deref());
        }
        _ => {
            // Full model: parse + scan per container, then emit json + table.
            let (out, ifmt, header_note) = match sniff.container {
                Container::Gguf => {
                    let ctx = match gguf::parse_gguf(&path) {
                        Ok(c) => c,
                        Err(e) => fatal(&e),
                    };
                    let out = match gguf::scan_gguf(&path, &ctx) {
                        Ok(o) => o,
                        Err(e) => fatal(&e),
                    };
                    let note = format!("gguf v{}, {} tensors", ctx.version, ctx.tensors.len());
                    (out, "gguf", note)
                }
                Container::Safetensors => {
                    let hdr = match formats::parse_header(&path) {
                        Ok(h) => h,
                        Err(e) => fatal(&e),
                    };
                    if hdr.is_adapter {
                        // sniff said FullModel but the header is a LoRA adapter
                        let json = adapter_json(&path, &hdr);
                        println!("theseus-scan  {}  PEFT/LoRA adapter", path);
                        let a = &hdr.adapter;
                        println!(
                            "  lora_A={} lora_B={} rank={} targets={} tensors={}",
                            a.has_lora_a,
                            a.has_lora_b,
                            a.rank.map(|r| r.to_string()).unwrap_or_else(|| "?".into()),
                            a.target_modules.join(","),
                            a.tensor_count
                        );
                        println!("  verdict: {}", formats::ADAPTER_VERDICT);
                        write_out(&json, json_out.as_deref());
                        exit(0);
                    }
                    let out = match formats::scan_safetensors(&path, &hdr) {
                        Ok(o) => o,
                        Err(e) => fatal(&e),
                    };
                    (out, "safetensors", String::new())
                }
                _ => fatal("unrecognized container"),
            };
            let tr = total_with_override(&out.per_family, &out.total);
            println!(
                "theseus-scan  {}  ({} 2-D weight tensors metered, {} skipped; {})",
                path, out.metered_tensors, out.skipped.len(), header_note
            );
            render_table(&out.per_family, &tr);
            let scheme = scheme_from_counts(&out.type_counts);
            println!(
                "quantization: scheme={} stats_source={} types={:?}",
                scheme.clone().unwrap_or_else(|| "null".into()),
                if scheme.is_some() { "structural" } else { "dequantized" },
                out.type_counts
            );
            let preflight = if mode == Mode::Preflight {
                let ops = ops_matrix(&out.per_family, &tr);
                println!("PREFLIGHT  operation x risk (thresholds provisional, n=2 measured contrast)");
                println!("{:<20} {:<12} {}", "operation", "verdict", "reason");
                for (op, ver, reason) in &ops {
                    println!("{:<20} {:<12} {}", op, ver, reason);
                }
                Some(ops)
            } else {
                None
            };
            let json = build_census_json(
                &path,
                &out.per_family,
                &tr,
                &out.skipped,
                &out.type_counts,
                ifmt,
                preflight.as_deref(),
            );
            write_out(&json, json_out.as_deref());
            let vds = verdicts(&out.per_family, &tr);
            println!("verdicts (thresholds provisional, calibrated on n=2 measured pairs):");
            if vds.is_empty() {
                println!("  none: no family exceeds the provisional quant/export/adaptation risk limits");
            } else {
                for v in &vds {
                    println!("  {}", v);
                }
            }
        }
    }
    eprint!("scan wall time: {:.2} s\n", t0.elapsed().as_secs_f64());
}

fn fatal(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    exit(1)
}

#[cfg(test)]
mod tests;
