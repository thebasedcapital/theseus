//! tests.rs — scoped self-checks for theseus-scan (owner: ScanFormats). No formatters/linters.
//!
//! Covers the census math (hand-computed block statistics, tail handling, f16/bf16 extremes),
//! container sniffing (PEFT adapter + GGUF round-trip on a synthetic 2-tensor file), and the
//! structural amax reconstruction for Q8_0 / Q4_K / Q5_K / Q6_K against hand-built blocks, plus
//! fail-closed sniffing of short/truncated files (never a wrong container, never a fake label).

use std::collections::BTreeMap;
use std::fs;

use crate::formats::{self, ArtifactKind, Container, Dtype};
use crate::gguf::{self, Blk, RowSink};
use crate::stats::{bf16_to_f32, f16_to_f32, StatAcc};

fn f16b(v: f32) -> u8 {
    // f16 -> 2 LE bytes for hand-building blocks (exact for the test values used: integers/powers).
    let h: u16 = if v == 1.0 {
        0x3C00
    } else if v == 0.0 {
        0
    } else {
        // minimal: recompute via bit math for the simple values used
        let neg = v < 0.0;
        let a = v.abs();
        let exp = a.log2().floor() as i32;
        let mant = (a / 2f32.powi(exp) - 1.0) * 1024.0;
        let e = (exp + 15).clamp(0, 31) as u16;
        let m = mant.round() as u16;
        let sign = if neg { 1u16 << 15 } else { 0 };
        sign | (e << 10) | (m & 0x3FF)
    };
    let bytes = h.to_le_bytes();
    bytes[0]
}

#[test]
fn block_statistic_64_ones_is_1_over_588() {
    // 64 ones = two full 32-blocks with amax 1: pooled = 2*(1^2*32)/(12*49*64) = 1/588,
    // i.e. the contract formula sum_blocks(amax^2*32)/(12*49*sum w^2).
    let mut a = StatAcc::new();
    a.feed_row(&vec![1.0f32; 64]);
    a.close_tensor();
    let r = a.report();
    assert!(
        (r["q4_block_mse_pooled"] - 1.0 / 588.0).abs() < 1e-12,
        "pooled"
    );
    assert!((r["q4_block_mse"] - 1.0 / 588.0).abs() < 1e-12, "mean");
    assert_eq!(r["weights"], 64.0);
    assert_eq!(r["below_f16_normal"], 0.0);
}

#[test]
fn short_tail_block_is_not_counted_as_a_full_block() {
    // 48 values: one full 32-block plus a 16-element tail that enters the denominator only.
    let mut v: Vec<f32> = vec![1.0; 32];
    v.extend_from_slice(&[0.5; 16]);
    let mut a = StatAcc::new();
    a.feed_row(&v);
    a.close_tensor();
    let r = a.report();
    let sum_sq = 32.0 + 16.0 * 0.25;
    let expect = (1.0f64 * 32.0) / (12.0 * 49.0 * sum_sq);
    assert!((r["q4_block_mse_pooled"] - expect).abs() < 1e-12);
    assert_eq!(r["weights"], 48.0);
}

#[test]
fn f16_bf16_decode_covers_subnormals_65504_inf_nan() {
    // mirrors inspect/src/main.rs tests
    assert_eq!(f16_to_f32(0x3c00), 1.0);
    assert_eq!(f16_to_f32(0x0001), 5.960464477539063e-8); // 2^-24 subnormal
    assert_eq!(f16_to_f32(0x0400), 6.103515625e-5); // smallest normal (2^-14)
    assert_eq!(f16_to_f32(0x7bff), 65504.0); // largest finite
    assert!(f16_to_f32(0x7c01).is_nan());
    assert!(f16_to_f32(0x7c00).is_infinite());
    assert_eq!(bf16_to_f32(0x3f80), 1.0);
    assert_eq!(bf16_to_f32(0x3880), 6.103515625e-5);
    assert_eq!(bf16_to_f32(0x0080), 1.1754943508222875e-38);
}

#[test]
fn adapter_sniff_and_metrics() {
    // Build a small PEFT-LoRA safetensors: q_proj lora_A + lora_B with rank r=8.
    let header = r#"{"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":{"dtype":"F32","shape":[8,16],"data_offsets":[0,512]},"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":{"dtype":"F32","shape":[16,8],"data_offsets":[512,1024]}}"#;
    let hs = header.as_bytes().len() as u64;
    let mut file: Vec<u8> = Vec::new();
    file.extend((hs as u64).to_le_bytes());
    file.extend(header.as_bytes());
    file.extend(vec![0u8; 1024]); // data area (unused by header-only adapter scan)
    let p = std::env::temp_dir().join("theseus_scan_adapter_test.safetensors");
    fs::write(&p, &file).unwrap();

    let sniff = formats::sniff(p.to_str().unwrap());
    assert_eq!(sniff.container, Container::Safetensors);
    let hdr = formats::parse_header(p.to_str().unwrap()).unwrap();
    assert!(hdr.is_adapter, "must be classified as an adapter");
    assert_eq!(hdr.adapter.rank, Some(8));
    assert!(hdr.adapter.target_modules.contains(&"q_proj".to_string()));
    assert_eq!(hdr.adapter.tensor_count, 2);
    assert_eq!(hdr.adapter.pair_count, 1);
    assert_eq!(hdr.adapter.dtype.as_deref(), Some("F32"));
    assert_eq!(
        formats::ADAPTER_VERDICT,
        "ADAPTER: static risk flags not defined for adapters yet"
    );
    fs::remove_file(&p).ok();
}

#[test]
fn short_or_truncated_files_stay_unsupported_not_mislabeled() {
    // An 11-byte .bin is classified by its extension (torch pickle), never "unreadable/empty".
    let p = std::env::temp_dir().join("theseus_scan_short_unsupported.bin");
    fs::write(&p, b"\x80\x02torch...").unwrap();
    let s = formats::sniff(p.to_str().unwrap());
    assert_eq!(s.container, Container::PyTorchBin);
    assert_eq!(s.kind, ArtifactKind::Unsupported);
    fs::remove_file(&p).ok();
    // A truncated safetensors (8 bytes of 0xff + '{') never classifies as a container.
    let q = std::env::temp_dir().join("theseus_scan_truncated.safetensors");
    let mut bytes = vec![0xFFu8; 8];
    bytes.push(b'{');
    fs::write(&q, &bytes).unwrap();
    let s = formats::sniff(q.to_str().unwrap());
    assert_eq!(s.container, Container::Unknown);
    assert_eq!(s.kind, ArtifactKind::Unsupported);
    fs::remove_file(&q).ok();
}

#[test]
fn moe_family_keying_is_boundary_aware_and_fail_closed() {
    assert_eq!(
        formats::family_of("model.layers.0.mlp.gate_proj.weight"),
        Some("gate_proj")
    );
    assert_eq!(
        formats::family_of("model.layers.0.mlp.experts.3.gate_proj.weight"),
        Some("expert_gate")
    );
    assert_eq!(
        formats::family_of("model.layers.0.mlp.experts.3.up_proj.weight"),
        Some("expert_up")
    );
    assert_eq!(
        formats::family_of("model.layers.0.mlp.experts.3.down_proj.weight"),
        Some("expert_down")
    );
    assert_eq!(
        formats::family_of("model.layers.0.mlp.experts.3.gate_up_proj.weight"),
        Some("__unavailable_expert_fused")
    );
    assert_eq!(
        gguf::gguf_family_of("blk.0.block_sparse_moe.experts.3.w1.weight"),
        Some("expert_gate")
    );
    assert_eq!(
        gguf::gguf_family_of("blk.0.block_sparse_moe.experts.3.w2.weight"),
        Some("expert_down")
    );
    assert_eq!(
        gguf::gguf_family_of("blk.0.block_sparse_moe.experts.3.w3.weight"),
        Some("expert_up")
    );
    assert_eq!(
        gguf::gguf_family_of("blk.0.ffn_gate_exps.weight"),
        Some("__unavailable_expert_fused")
    );
    assert_eq!(
        gguf::gguf_family_of("blk.0.ffn_down_exps.weight"),
        Some("__unavailable_expert_fused")
    );
    assert_eq!(
        gguf::gguf_family_of("blk.0.ffn_up_exps.weight"),
        Some("__unavailable_expert_fused")
    );
    assert_eq!(
        gguf::gguf_family_of("blk.0.ffn_gate.3.weight"),
        Some("expert_gate")
    );
    assert_eq!(
        gguf::gguf_family_of("blk.0.ffn_down.3.weight"),
        Some("expert_down")
    );
    assert_eq!(
        gguf::gguf_family_of("blk.0.ffn_up.3.weight"),
        Some("expert_up")
    );
}

#[test]
fn q8_v3_threshold_is_operation_specific() {
    let mut acc = StatAcc::new();
    let mut row = vec![1.0f32; 32];
    row[0] = 3.28;
    acc.feed_row(&row);
    acc.close_tensor();
    let mut fam = BTreeMap::new();
    fam.insert("q_proj", acc);
    let total = fam["q_proj"].report();
    let ops = crate::ops_matrix(&fam, &total);
    assert_eq!(
        ops.iter().find(|x| x.0 == "quantize.gguf.q8_0").unwrap().1,
        "AT_RISK"
    );
    assert_eq!(
        ops.iter()
            .find(|x| x.0 == "quantize.gguf.q4_k_m")
            .unwrap()
            .1,
        "OK"
    );
}
// ---- synthetic GGUF builder (in-test write side of the format) ----
struct Gb {
    v: Vec<u8>,
}
impl Gb {
    fn str(&mut self, s: &str) {
        self.v.extend((s.len() as u64).to_le_bytes());
        self.v.extend(s.as_bytes());
    }
    fn u32(&mut self, x: u32) {
        self.v.extend(x.to_le_bytes());
    }
    fn u64(&mut self, x: u64) {
        self.v.extend(x.to_le_bytes());
    }
    fn pad(&mut self, alignment: u64) {
        while (self.v.len() as u64) % alignment != 0 {
            self.v.push(0);
        }
    }
}

fn build_synthetic_gguf() -> Vec<u8> {
    // version 3, 1 KV (general.alignment=32), 2 tensors: attn_q F16[32,4], ffn_gate F32[16,2].
    let mut g = Gb { v: Vec::new() };
    g.v.extend(b"GGUF");
    g.u32(3);
    g.u64(2); // n_tensors
    g.u64(1); // n_kv
    g.str("general.alignment");
    g.u32(4); // GGUF_TYPE_UINT32
    g.u32(32);
    // tensor 1: F16 [32,4] -> 256 bytes, offset 0
    g.str("blk.0.attn_q.weight");
    g.u32(2);
    g.u64(32);
    g.u64(4);
    g.u32(1); // F16
    g.u64(0);
    // tensor 2: F32 [16,2] -> 128 bytes, offset 256 (padded to 32)
    g.str("blk.0.ffn_gate.weight");
    g.u32(2);
    g.u64(16);
    g.u64(2);
    g.u32(0); // F32
    g.u64(256);
    g.pad(32);
    // data: attn_q all 1.0 f16 (0x3C00); ffn_gate all 2.0 f32
    // each f16 element is 2 bytes; 32*4 = 128 elements -> 256 bytes
    for _ in 0..(32 * 4) {
        g.v.extend([0x00, 0x3C]);
    }
    for _ in 0..(16 * 2) {
        g.v.extend(2.0f32.to_le_bytes());
    }
    g.v
}

#[test]
fn gguf_header_roundtrip_and_f16_scan() {
    let bytes = build_synthetic_gguf();
    let p = std::env::temp_dir().join("theseus_scan_gguf_test.gguf");
    fs::write(&p, &bytes).unwrap();

    let ctx = gguf::parse_gguf(p.to_str().unwrap()).unwrap();
    assert_eq!(ctx.version, 3);
    assert_eq!(ctx.tensors.len(), 2);
    assert_eq!(ctx.n_kv, 1);
    assert_eq!(ctx.alignment, 32);
    assert!(ctx.data_start % 32 == 0, "data starts aligned");
    // attn_q data begins exactly at data_start; ffn_gate at data_start+256 (padded)
    assert_eq!(ctx.tensors[0].offset, 0);
    assert_eq!(ctx.tensors[1].offset, 256);
    assert_eq!(ctx.tensors[0].nbytes, 32 * 4 * 2);
    assert_eq!(ctx.tensors[1].nbytes, 16 * 2 * 4);

    let out = gguf::scan_gguf(p.to_str().unwrap(), &ctx).unwrap();
    assert_eq!(out.metered_tensors, 2);
    assert_eq!(out.per_family.len(), 2);
    // attn_q = 128 ones -> four full 32-blocks of ones -> q4_block_mse ratio 1/588
    let q = out.per_family["q_proj"].report();
    assert!((q["q4_block_mse"] - 1.0 / 588.0).abs() < 1e-12);
    assert!((q["q4_block_mse_pooled"] - 1.0 / 588.0).abs() < 1e-12);
    assert_eq!(q["weights"], 128.0);
    // ffn_gate [16,2]: every row is a 16-element tail (< the 32-census block), so it enters the
    // denominator only — both q4 conventions are 0 (the same short-tail contract as feed_row).
    let q = out.per_family["gate_proj"].report();
    assert_eq!(q["q4_block_mse"], 0.0);
    assert_eq!(q["q4_block_mse_pooled"], 0.0);
    assert_eq!(q["weights"], 32.0);
    fs::remove_file(&p).ok();
}

#[test]
fn q8_0_hand_block_amax_reconstruction() {
    // Q8_0: v = d*qs, amax = d*max|qs| exactly (block_q8_0, ggml-common.h:244-246).
    let mut blk_bytes: Vec<u8> = Vec::new();
    blk_bytes.push(0x00);
    blk_bytes.push(0x3C); // d = 1.0 f16 LE
    let mut qs = vec![0i8; 32];
    qs[0] = -64;
    qs[1] = 127;
    qs[2] = 127;
    qs[3] = 1;
    qs[31] = 2;
    for q in qs {
        blk_bytes.push(q as u8);
    }
    assert_eq!(blk_bytes.len(), 34);

    let mut acc = StatAcc::new();
    {
        let mut sink = RowSink::new(&mut acc);
        gguf::scan_q8_0(&blk_bytes, &mut sink);
        sink.end_row();
    }
    acc.close_tensor();
    let r = acc.report();
    eprintln!("q8_0 sum_sq={} amax={} n={}", acc.sum_sq, acc.amax, acc.n);
    // structural amax = d*max|qs| = 1*127
    let amax = blk_bytes[2..34]
        .iter()
        .map(|&q| ((q as i8) as i32).abs())
        .max()
        .unwrap() as f64;
    assert!((r["dyn_range_log10"] - 127.0f64.log10()).abs() < 1e-9); // amax 127 / amin 1
                                                                     // energy = sum qs^2 (d=1)
    let sum_sq: f64 = blk_bytes[2..34]
        .iter()
        .map(|&q| (q as i8 as f64).powi(2))
        .sum();
    assert!((r["amax_over_rms"] * (sum_sq / 32.0).sqrt() - amax).abs() < 1e-9);
    assert!(
        (r["q4_block_mse_pooled"] - (amax * amax * 32.0) / (12.0 * 49.0 * sum_sq)).abs() < 1e-12
    );
    assert_eq!(r["weights"], 32.0);
}

#[test]
fn q4_k_hand_superblock_ones() {
    // One Q4_K super-block (256 elts): d=1.0, dmin=0, every scale index =63, every q nibble =15
    // -> every value = 1*63*15 - 0 = 945. amax 945, energy 256*945^2, ratio 1/588.
    let mut b: Vec<u8> = Vec::new();
    b.push(0x00);
    b.push(0x3C); // d = 1.0 f16 LE
    b.push(0x00);
    b.push(0x00); // dmin = 0.0
                  // scales[12] with sc=63 (packing per printf in quantize_row_q4_K_impl / get_scale_min_k4)
    b.extend([0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0x0F, 0x0F, 0x0F, 0x0F]);
    b.extend(vec![0xFFu8; 128]); // all nibbles 15
    assert_eq!(b.len(), 144);
    // verify the unpack agrees: sub-block 0 scale = 63, min = 0
    let (sc, m) = gguf::scale_min_k4(0, &b[4..16]);
    assert_eq!((sc, m), (63, 0));
    let (sc, m) = gguf::scale_min_k4(7, &b[4..16]);
    assert_eq!((sc, m), (63, 0));

    let mut acc = StatAcc::new();
    {
        let mut sink = RowSink::new(&mut acc);
        gguf::scan_q4_k(&b, &mut sink);
        sink.end_row();
    }
    acc.close_tensor();
    let r = acc.report();
    assert!((r["amax_over_rms"] * (945.0f64 * 945.0f64).sqrt() - 945.0).abs() < 1e-6);
    assert!((r["q4_block_mse_pooled"] - 1.0 / 588.0).abs() < 1e-12);
    assert_eq!(r["weights"], 256.0);
    // both conventions agree on a single tensor
    assert!((r["q4_block_mse"] - r["q4_block_mse_pooled"]).abs() < 1e-12);
}

#[test]
fn q5_k_and_q6_k_hand_amax() {
    // Q5_K: d=1, dmin=0, scales sc=63/min=0 (same packing as q4_K), qh[0] bit0=1 high bit for
    // sub-block 0 element 0 -> that element q = 15 + 16 = 31 -> v = 63*31 = 1953.
    let mut b5: Vec<u8> = Vec::new();
    b5.extend([0x00, 0x3C, 0x00, 0x00]); // d=1, dmin=0
    b5.extend([0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0x0F, 0x0F, 0x0F, 0x0F]);
    let mut qh = vec![0u8; 32];
    qh[0] = 0x01; // sub-block 0 (is=0), element 0 high bit
    b5.extend(qh);
    let mut qs = vec![0xFFu8; 128];
    // element 0 keeps q = 15 + 16 (high bit set) = 31 -> v max = 63*31 = 1953.
    b5.extend(qs);
    assert_eq!(b5.len(), 176);

    let mut acc = StatAcc::new();
    {
        let mut sink = RowSink::new(&mut acc);
        gguf::scan_q5_k(&b5, &mut sink);
        sink.end_row();
    }
    acc.close_tensor();
    let r = acc.report();
    // block 0: element0 v=63*31=1953; amax across block = 1953; energy includes 1*1953^2 + 255*(945^2)
    let sum_sq = 1953.0 * 1953.0 + 255.0 * 945.0 * 945.0;
    assert!((r["amax_over_rms"] * (sum_sq / 256.0f64).sqrt() - 1953.0).abs() < 1e-9);
    assert!(
        (r["q4_block_mse_pooled"]
            - (32.0 * (1953.0f64.powi(2) + 7.0 * 945.0f64.powi(2))) / (12.0 * 49.0 * sum_sq))
            .abs()
            < 5e-6
    );

    // Q6_K: d=1, all scales =1, ql high/low nibbles = 2, qh=0 -> q = 2 - 32 = -30, v=-30.
    let mut b6: Vec<u8> = Vec::new();
    b6.extend(vec![0x22u8; 128]); // ql: low nibble 2, high nibble 2
    b6.extend(vec![0u8; 64]); // qh
    b6.extend(vec![1i8 as u8; 16]); // scales all 1
    b6.extend([0x00, 0x3C]); // d = 1.0
    assert_eq!(b6.len(), 210);
    let mut acc6 = StatAcc::new();
    {
        let mut sink = RowSink::new(&mut acc6);
        gguf::scan_q6_k(&b6, &mut sink);
        sink.end_row();
    }
    acc6.close_tensor();
    let r6 = acc6.report();
    let sum6 = 256.0 * 30.0 * 30.0;
    assert!((r6["amax_over_rms"] * (sum6 / 256.0f64).sqrt() - 30.0).abs() < 1e-9);
    assert!((r6["q4_block_mse_pooled"] - 1.0 / 588.0).abs() < 5e-6); // 8 equal blocks, scale-invariant
    assert_eq!(r6["weights"], 256.0);
}

#[test]
fn structural_amax_equals_full_scan_on_randomish_blocks() {
    // For each structural scanner, the per-block reconstructed amax must equal max|dequantized v|
    // computed element-by-element (the "if I did the full dequant" oracle). Deterministic PRNG.
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    // q4_K: random scales + random nibbles
    let mut b4: Vec<u8> = Vec::new();
    b4.extend([0x00, 0x3C, 0x00, 0x00]);
    for i in 0..12 {
        b4.push((rnd() & 0xFF) as u8);
    }
    for i in 0..128 {
        b4.push((rnd() & 0xFF) as u8);
    }
    let mut acc = StatAcc::new();
    {
        let mut sink = RowSink::new(&mut acc);
        gguf::scan_q4_k(&b4, &mut sink);
        sink.end_row();
    }
    // oracle: full dequant per dequantize_row_q4_K (ggml-quants.c:1471-1488)
    let d = f16_to_f32(u16::from_le_bytes([b4[0], b4[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([b4[2], b4[3]]));
    let mut oracle_amax: f64 = 0.0;
    for is in 0..8usize {
        let (sc, m) = gguf::scale_min_k4(is, &b4[4..16]);
        let dl = d * sc as f32;
        let ml = dmin * m as f32;
        for l in 0..32 {
            let q = gguf::qs_nibble(&b4[16..144], is, l) as i32;
            let v = (dl * q as f32 - ml) as f64;
            if v.abs() > oracle_amax {
                oracle_amax = v.abs();
            }
        }
    }
    let struct_amax = acc.amax;
    assert!(
        ((struct_amax - oracle_amax) / oracle_amax).abs() < 1e-12,
        "q4_k structural amax {} vs oracle {}",
        struct_amax,
        oracle_amax
    );
}

#[test]
fn scheme_from_counts_and_unsupported_types() {
    // single quant type -> canonical scheme; float-only -> None (scheme null in JSON)
    let mut m: BTreeMap<&'static str, usize> = BTreeMap::new();
    m.insert("q4_k", 40);
    m.insert("q6_k", 4);
    m.insert("f16", 2);
    assert!(crate::scheme_from_counts(&m).unwrap().starts_with("mixed:"));
    m.clear();
    m.insert("q4_k", 42);
    assert_eq!(crate::scheme_from_counts(&m).as_deref(), Some("q4_k_m"));
    m.clear();
    m.insert("f16", 50);
    assert!(crate::scheme_from_counts(&m).is_none());
    // unsupported grid types map to GType::Unsupported and never fake a scan
    assert!(!gguf::GType::from_code(21).supported()); // iq3_s
    assert!(gguf::GType::from_code(12).supported()); // q4_k
}
