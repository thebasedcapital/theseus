//! gguf.rs — GGUF container parse + block-quantized structural statistics (owner: ScanFormats).
//!
//! Parses the GGUF header (magic/version/tensor-count/KV/tensor-meta; ggml/src/gguf.cpp:539-757),
//! then streams each metered weight tensor row by row. Every block-quant type is scanned
//! STRUCTURALLY — per-census-32-block extrema are reconstructed from the stored scale/q bits per
//! dequantize_row_* (ggml/src/ggml-quants.c), statistics accumulate inline, and no tensor is ever
//! materialized. Float types ride the dequantized `feed_row` path.
//!
//! per-type amax reconstruction (format -> formula -> source):
//!   Q8_0  v = d·qs             amax = |d|·max|qs|            ggml-common.h:244-246; ggml-quants.c:495-504
//!   Q4_K  v = d·sc·q − dmin·m  6-bit sc/m (sc∈[0,63], q∈[0,15], affine in q) amax from q min/max,
//!        divisor 63 folded into d at quant time (d = max_scale/63)  ggml-quants.c:822-834,1471-1488
//!   Q5_K  like Q4_K but q∈[0,31] with 5th bit in qh              ggml-quants.c:1673-1704
//!   Q6_K  v = d·sc[i]·q, q∈[−32,31], amax=|d·sc[i]|·max|q|       ggml-quants.c:1881-1908
//!   IQ4_XS dl = d·(ls−32) with 6-bit ls; v=dl·kvalues_iq4nl[nib] ggml-quants.c:2671-2697; ggml-common.h:1110-1112

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::formats::ScanOut;
use crate::stats::{StatAcc, bf16_to_f32, f16_to_f32, F16_NORMAL_MIN};

pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

// kvalues_iq4nl — ggml-common.h:1110-1112 (int8 non-linear 4-bit lattice values).
const KVALUES_IQ4NL: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GType {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q1_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    Iq4Nl,
    Iq4Xs,
    Unsupported(u32),
}

impl GType {
    pub(crate) fn from_code(c: u32) -> GType {
        match c {
            0 => GType::F32,
            1 => GType::F16,
            30 => GType::Bf16,
            8 => GType::Q8_0,
            2 => GType::Q4_0,
            3 => GType::Q4_1,
            6 => GType::Q5_0,
            7 => GType::Q5_1,
            41 => GType::Q1_0,
            10 => GType::Q2_K,
            11 => GType::Q3_K,
            12 => GType::Q4_K,
            13 => GType::Q5_K,
            14 => GType::Q6_K,
            15 => GType::Q8_K,
            20 => GType::Iq4Nl,
            23 => GType::Iq4Xs,
            other => GType::Unsupported(other),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            GType::F32 => "f32",
            GType::F16 => "f16",
            GType::Bf16 => "bf16",
            GType::Q8_0 => "q8_0",
            GType::Q4_0 => "q4_0",
            GType::Q4_1 => "q4_1",
            GType::Q5_0 => "q5_0",
            GType::Q5_1 => "q5_1",
            GType::Q1_0 => "q1_0",
            GType::Q2_K => "q2_k",
            GType::Q3_K => "q3_k",
            GType::Q4_K => "q4_k",
            GType::Q5_K => "q5_k",
            GType::Q6_K => "q6_k",
            GType::Q8_K => "q8_k",
            GType::Iq4Nl => "iq4_nl",
            GType::Iq4Xs => "iq4_xs",
            GType::Unsupported(_) => "unsupported",
        }
    }
    /// quantization scheme for `quantization.scheme` when this type dominates.
    pub fn supported(&self) -> bool {
        !matches!(self, GType::Unsupported(_))
    }
    pub fn scheme(&self) -> &'static str {
        match self {
            GType::Q8_0 => "q8_0",
            GType::Q4_0 => "q4_0",
            GType::Q4_1 => "q4_1",
            GType::Q5_0 => "q5_0",
            GType::Q5_1 => "q5_1",
            GType::Q1_0 => "q1_0",
            GType::Q2_K => "q2_k",
            GType::Q3_K => "q3_k",
            GType::Q4_K => "q4_k_m",
            GType::Q5_K => "q5_k_m",
            GType::Q6_K => "q6_k",
            GType::Q8_K => "q8_k",
            GType::Iq4Nl => "iq4_nl",
            GType::Iq4Xs => "iq4_xs",
            _ => "mixed",
        }
    }
    /// elements per quant unit and bytes per quant unit (ggml.c type_traits / ggml-common.h structs)
    pub fn blk(&self) -> (usize, usize) {
        match self {
            GType::F32 => (1, 4),
            GType::F16 | GType::Bf16 => (1, 2),
            GType::Q8_0 => (32, 34),
            GType::Q4_0 => (32, 18),
            GType::Q4_1 => (32, 20),
            GType::Q5_0 => (32, 22),
            GType::Q5_1 => (32, 24),
            GType::Q1_0 => (128, 18),
            GType::Q2_K => (256, 84),
            GType::Q3_K => (256, 110),
            GType::Q4_K => (256, 144),
            GType::Q5_K => (256, 176),
            GType::Q6_K => (256, 210),
            GType::Q8_K => (256, 292),
            GType::Iq4Nl => (32, 18),
            GType::Iq4Xs => (256, 136),
            GType::Unsupported(_) => (0, 0),
        }
    }
}

pub struct TensorMeta {
    pub name: String,
    pub ne: [u64; 4],
    pub n_dims: usize,
    pub ty: GType,
    pub offset: u64, // relative to data section start
    pub nbytes: u64,
}

pub struct GgufCtx {
    pub version: u32,
    pub alignment: u64,
    pub data_start: u64, // absolute file offset of the (aligned) data section
    pub tensors: Vec<TensorMeta>,
    pub n_kv: u64,
    pub arch: Option<String>,
    pub name: Option<String>,
    pub file_type: Option<u32>,
}

fn rd_u32(f: &mut impl Read) -> Result<u32, String> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b).map_err(|e| format!("read u32: {}", e))?;
    Ok(u32::from_le_bytes(b))
}
fn rd_u64(f: &mut impl Read) -> Result<u64, String> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).map_err(|e| format!("read u64: {}", e))?;
    Ok(u64::from_le_bytes(b))
}
fn rd_str(f: &mut impl Read) -> Result<String, String> {
    let n = rd_u64(f)?;
    if n > (1 << 30) {
        return Err(format!("implausible string length {}", n));
    }
    let mut s = vec![0u8; n as usize];
    f.read_exact(&mut s).map_err(|e| format!("read string: {}", e))?;
    Ok(String::from_utf8_lossy(&s).into_owned())
}

/// Parse the GGUF header. Layout (write side gguf.cpp:1459-1629, read side 539-757):
/// magic "GGUF" | u32 version | u64 n_tensors | u64 n_kv | KVs | tensor metas | pad->alignment.
pub fn parse_gguf(path: &str) -> Result<GgufCtx, String> {
    let f = File::open(path).map_err(|e| e.to_string())?;
    let mut br = BufReader::with_capacity(1 << 16, f);
    let mut magic = [0u8; 4];
    br.read_exact(&mut magic).map_err(|e| format!("read magic: {}", e))?;
    if &magic != b"GGUF" {
        return Err("not a GGUF file (bad magic)".into());
    }
    let version = rd_u32(&mut br)?;
    if version != 2 && version != 3 {
        return Err(format!("unsupported GGUF version {}", version));
    }
    let n_tensors = rd_u64(&mut br)?;
    let n_kv = rd_u64(&mut br)?;
    if n_tensors > (1 << 26) || n_kv > (1 << 26) {
        return Err("implausible GGUF counts".into());
    }

    let mut alignment = GGUF_DEFAULT_ALIGNMENT;
    let mut arch: Option<String> = None;
    let mut name: Option<String> = None;
    let mut file_type: Option<u32> = None;

    // Value sizes per GGUF type (bytes): 0,1->1; 2,3->2; 4,5,6,7->4; 8->string; 9->array; 10,11,12->8.
    for _ in 0..n_kv {
        let key = rd_str(&mut br)?;
        let t = rd_u32(&mut br)?;
        let value_u32 = |br: &mut BufReader<File>| -> Result<u32, String> {
            let mut b = [0u8; 4];
            br.read_exact(&mut b).map_err(|e| format!("read value: {}", e))?;
            Ok(u32::from_le_bytes(b))
        };
        match t {
            0 | 1 => {
                let mut b = [0u8; 1];
                br.read_exact(&mut b).map_err(|e| format!("read u8: {}", e))?;
            }
            2 | 3 => {
                let mut b = [0u8; 2];
                br.read_exact(&mut b).map_err(|e| format!("read u16: {}", e))?;
            }
            4 | 5 | 6 => {
                // GGUF type 4/5/6 (UINT32/INT32/FLOAT32): 4-byte scalars.
                let v = value_u32(&mut br)?;
                match key.as_str() {
                    "general.alignment" => {
                        alignment = v as u64;
                        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
                            return Err(format!("alignment {} is not a power of 2", alignment));
                        }
                    }
                    "general.file_type" => file_type = Some(v),
                    _ => {}
                }
            }
            7 => {
                // GGUF_TYPE_BOOL is 1 byte, NOT 4 (gguf-py gguf_reader.py:129 BOOL -> np.bool_;
                // gguf_writer.py:1392 packs BOOL with the '?' fmt). Reading 4 here would
                // desync every subsequent KV on real llama.cpp files.
                let mut b = [0u8; 1];
                br.read_exact(&mut b).map_err(|e| format!("read bool: {}", e))?;
            }
            8 => {
                let s = rd_str(&mut br)?;
                match key.as_str() {
                    "general.architecture" => arch = Some(s),
                    "general.name" => name = Some(s),
                    _ => {}
                }
            }
            9 => {
                // array: elem type (u32), count (u64), then `count` element payloads.
                let et = rd_u32(&mut br)?;
                let n = rd_u64(&mut br)?;
                for _ in 0..n {
                    match et {
                        0 | 1 => {
                            let mut b = [0u8; 1];
                            br.read_exact(&mut b).map_err(|e| format!("arr u8: {}", e))?;
                        }
                        2 | 3 => {
                            let mut b = [0u8; 2];
                            br.read_exact(&mut b).map_err(|e| format!("arr u16: {}", e))?;
                        }
                        4 | 5 | 6 => {
                            let _ = value_u32(&mut br)?;
                        }
                        7 => {
                            // BOOL array elements are also 1 byte (see the scalar arm note).
                            let mut b = [0u8; 1];
                            br.read_exact(&mut b).map_err(|e| format!("arr bool: {}", e))?;
                        }
                        8 => {
                            let _ = rd_str(&mut br)?;
                        }
                        10 | 11 | 12 => {
                            let _ = rd_u64(&mut br)?;
                        }
                        other => return Err(format!("bad array elem type {}", other)),
                    }
                }
            }
            10 | 11 | 12 => {
                let _ = rd_u64(&mut br)?;
            }
            other => return Err(format!("bad GGUF value type {}", other)),
        }
    }

    let mut tensors: Vec<TensorMeta> = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let tname = rd_str(&mut br)?;
        let n_dims = rd_u32(&mut br)? as usize;
        if n_dims > 4 {
            return Err(format!("tensor '{}' has {} dims > 4", tname, n_dims));
        }
        let mut ne = [1u64; 4];
        for j in 0..n_dims {
            ne[j] = rd_u64(&mut br)?;
        }
        let code = rd_u32(&mut br)?;
        let ty = GType::from_code(code);
        let offset = rd_u64(&mut br)?;
        let (blk_el, blk_by) = ty.blk();
        let n_el: u64 = ne[0..n_dims].iter().product();
        let nbytes = if blk_el > 0 { n_el / blk_el as u64 * blk_by as u64 } else { 0 };
        tensors.push(TensorMeta {
            name: tname,
            ne,
            n_dims,
            ty,
            offset,
            nbytes,
        });
    }

    // data section starts aligned (gguf.cpp:751-757).
    let hdr_end = br.stream_position().unwrap_or(0);
    let aligned = if tensors.is_empty() {
        hdr_end
    } else {
        (hdr_end + alignment - 1) / alignment * alignment
    };
    Ok(GgufCtx {
        version,
        alignment,
        data_start: aligned,
        tensors,
        n_kv,
        arch,
        name,
        file_type,
    })
}

/// True GGUF family mapping for llama-family weight tensor names (blk.N.attn_* / ffn_*).
pub fn gguf_family_of(name: &str) -> Option<&'static str> {
    if name.contains("attn_qkv") || name.contains("attn_kv") {
        return None; // fused projections: single family, different meaning
    }
    if name.contains("attn_output") {
        return Some("o_proj");
    }
    if name.contains("attn_q") {
        return Some("q_proj");
    }
    if name.contains("attn_k") {
        return Some("k_proj");
    }
    if name.contains("attn_v") {
        return Some("v_proj");
    }
    if name.contains("ffn_gate") {
        return Some("gate_proj");
    }
    if name.contains("ffn_up") {
        return Some("up_proj");
    }
    if name.contains("ffn_down") {
        return Some("down_proj");
    }
    None
}

/// Per-census-32-block structural accumulator emitted by scan_row_*.
#[derive(Clone, Copy)]
pub struct Blk {
    pub amax: f64,
    pub sum_sq: f64,
    pub below: u64,
    pub amin: f64,
    pub cnt: usize,
}
impl Default for Blk {
    /// amin is a running min over nonzero magnitudes; a 0.0 seed would never update it
    /// (reference inspector/news for StatAcc.amin_nz uses f64::INFINITY).
    fn default() -> Blk {
        Blk {
            amax: 0.0,
            sum_sq: 0.0,
            below: 0,
            amin: f64::INFINITY,
            cnt: 0,
        }
    }
}
impl Blk {
    #[inline]
    fn add(&mut self, v: f32) {
        let a = (v as f64).abs();
        self.cnt += 1;
        self.sum_sq += a * a;
        if a > 0.0 {
            if a < self.amin {
                self.amin = a;
            }
            if a < F16_NORMAL_MIN as f64 {
                self.below += 1;
            }
        }
    }
    #[inline]
    fn finish_amax(&mut self, a: f64) {
        if a > self.amax {
            self.amax = a;
        }
    }
}

/// Row sink: folds census-32-blocks into the StatAcc and keeps the row's L2 energy.
pub struct RowSink<'a> {
    pub acc: &'a mut StatAcc,
    row_sq: f64,
}
impl<'a> RowSink<'a> {
    pub fn new(acc: &'a mut StatAcc) -> RowSink<'a> {
        RowSink { acc, row_sq: 0.0 }
    }
    #[inline]
    pub(crate) fn block(&mut self, b: &Blk) {
        self.row_sq += b.sum_sq;
        self.acc.feed_block(b.amax, b.sum_sq, b.below, b.amin, b.cnt);
    }
    pub(crate) fn end_row(&mut self) {
        self.acc.row_energy(self.row_sq);
        self.row_sq = 0.0;
    }
}

#[inline]
fn u16le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

/// get_scale_min_k4 — ggml-quants.c:822-834: 6-bit scale + 6-bit min per 32-elt sub-block.
#[inline]
pub(crate) fn scale_min_k4(j: usize, scales: &[u8]) -> (u32, u32) {
    if j < 4 {
        ((scales[j] & 63) as u32, (scales[j + 4] & 63) as u32)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d as u32, m as u32)
    }
}

/// nibble (4-bit) value of (sub-block `is`, element `l`): low nibbles first half, high second.
#[inline]
pub(crate) fn qs_nibble(qs: &[u8], is: usize, l: usize) -> u32 {
    let byte = qs[32 * (is / 2) + l];
    if is % 2 == 0 {
        (byte & 0xF) as u32
    } else {
        (byte >> 4) as u32
    }
}

// --- row scanners: one per supported block type, all structural (no f32 array materialization) ---

pub(crate) fn scan_q8_0(buf: &[u8], sink: &mut RowSink) {
    // block_q8_0: f16 d + 32 i8 qs (ggml-common.h:244-246); v = d*qs (ggml-quants.c:495-504).
    for b in buf.chunks_exact(34) {
        let d = f16_to_f32(u16le(b));
        let mut blk = Blk::default();
        let mut maxq: f64 = 0.0;
        for &q in &b[2..34] {
            let a = (q as i8 as i32).abs() as f64;
            if a > maxq {
                maxq = a;
            }
            blk.add(d * q as i8 as f32);
        }
        blk.finish_amax((d as f64).abs() * maxq); // amax = d*max|qs| exactly
        sink.block(&blk);
    }
}

pub(crate) fn scan_q8_k(buf: &[u8], sink: &mut RowSink) {
    // block_q8_K: f32 d + 256 i8 qs + 32 i16 bsums (ggml-common.h:360-366), v = d*qs.
    for b in buf.chunks_exact(292) {
        let d = f32::from_le_bytes(b[0..4].try_into().unwrap());
        let mut blk = Blk::default();
        let mut maxq: f64 = 0.0;
        for &q in &b[4..260] {
            let a = (q as i8 as i32).abs() as f64;
            if a > maxq {
                maxq = a;
            }
            blk.add(d * q as i8 as f32);
        }
        blk.finish_amax((d as f64).abs() * maxq);
        sink.block(&blk);
    }
}

pub(crate) fn scan_q4_0(buf: &[u8], sink: &mut RowSink) {
    // v = d*(nib-8), nib∈[0,15] (ggml-quants.c:437-449); amax from nibble min/max.
    for b in buf.chunks_exact(18) {
        let d = f16_to_f32(u16le(b));
        let mut blk = Blk::default();
        let mut nmin = 16i32;
        let mut nmax = 0i32;
        for j in 0..16 {
            let byte = b[2 + j];
            let a = (byte & 0xF) as i32;
            let bb = (byte >> 4) as i32;
            nmin = nmin.min(a).min(bb);
            nmax = nmax.max(a).max(bb);
            blk.add(d * (a as f32 - 8.0));
            blk.add(d * (bb as f32 - 8.0));
        }
        let m = ((nmin as f64 - 8.0).abs()).max((nmax as f64 - 8.0).abs());
        blk.finish_amax((d as f64).abs() * m);
        sink.block(&blk);
    }
}

pub(crate) fn scan_q4_1(buf: &[u8], sink: &mut RowSink) {
    // v = d*nib + m (ggml-quants.c:457-468); affine in nib → amax at nibble min/max.
    for b in buf.chunks_exact(20) {
        let d = f16_to_f32(u16le(b));
        let m = f16_to_f32(u16le(&b[2..4]));
        let mut blk = Blk::default();
        let mut nmin = 16i32;
        let mut nmax = 0i32;
        for j in 0..16 {
            let byte = b[4 + j];
            let a = (byte & 0xF) as i32;
            let bb = (byte >> 4) as i32;
            nmin = nmin.min(a).min(bb);
            nmax = nmax.max(a).max(bb);
            blk.add(d * a as f32 + m);
            blk.add(d * bb as f32 + m);
        }
        let vlo = (d * nmin as f32 + m) as f64;
        let vhi = (d * nmax as f32 + m) as f64;
        blk.finish_amax(vlo.abs().max(vhi.abs()));
        sink.block(&blk);
    }
}

/// q5_0/q5_1 helper: the two 5-bit values in byte `word` (ggml-quants.c:476-488, 523-536).
#[inline]
fn q5_pair(qh: u32, word: u8, j: usize) -> (i32, i32) {
    let low = (word & 0xF) as i32;
    let high = (word >> 4) as i32;
    let b0 = ((qh >> (j + 0)) & 1) as i32; // element 2j: qh bit j
    let b1 = ((qh >> (j + 12)) & 1) as i32; // element 2j+1: qh bit j+12
    (low | (b0 << 4), high | (b1 << 4))
}

pub(crate) fn scan_q5_0(buf: &[u8], sink: &mut RowSink) {
    // v = d*(q5-16), q5∈[0,31] (ggml-quants.c:476-488); block amax = |d|*max|q5-16|.
    for b in buf.chunks_exact(22) {
        let d = f16_to_f32(u16le(b));
        let qh = u32::from_le_bytes(b[2..6].try_into().unwrap());
        let mut blk = Blk::default();
        let mut qmin = 32i32;
        let mut qmax = 0i32;
        for j in 0..16 {
            let (e0, e1) = q5_pair(qh, b[6 + j], j);
            qmin = qmin.min(e0).min(e1);
            qmax = qmax.max(e0).max(e1);
            blk.add(d * (e0 as f32 - 16.0));
            blk.add(d * (e1 as f32 - 16.0));
        }
        let m = ((qmin as f64 - 16.0).abs()).max((qmax as f64 - 16.0).abs());
        blk.finish_amax((d as f64).abs() * m);
        sink.block(&blk);
    }
}

pub(crate) fn scan_q5_1(buf: &[u8], sink: &mut RowSink) {
    // v = d*q5 + m (ggml-quants.c:523-536); affine → extremes at q5 min/max.
    for b in buf.chunks_exact(24) {
        let d = f16_to_f32(u16le(b));
        let m = f16_to_f32(u16le(&b[2..4]));
        let qh = u32::from_le_bytes(b[4..8].try_into().unwrap());
        let mut blk = Blk::default();
        let mut qmin = 32i32;
        let mut qmax = 0i32;
        for j in 0..16 {
            let (e0, e1) = q5_pair(qh, b[8 + j], j);
            qmin = qmin.min(e0).min(e1);
            qmax = qmax.max(e0).max(e1);
            blk.add(d * e0 as f32 + m);
            blk.add(d * e1 as f32 + m);
        }
        let vlo = (d * qmin as f32 + m) as f64;
        let vhi = (d * qmax as f32 + m) as f64;
        blk.finish_amax(vlo.abs().max(vhi.abs()));
        sink.block(&blk);
    }
}

pub(crate) fn scan_q1_0(buf: &[u8], sink: &mut RowSink) {
    // v = bit ? d : -d per 128-element block (ggml-quants.c:396-408); amax always |d|.
    for b in buf.chunks_exact(18) {
        let d = f16_to_f32(u16le(b));
        let mut blk = Blk::default();
        for &byte in &b[2..18] {
            for bit in 0..8 {
                let v = if (byte >> bit) & 1 == 1 { d } else { -d };
                blk.add(v);
            }
        }
        blk.finish_amax((d as f64).abs());
        sink.block(&blk);
    }
}

pub(crate) fn scan_q4_k(buf: &[u8], sink: &mut RowSink) {
    // block_q4_K: d,dmin f16 | scales[12] | qs[128] (ggml-common.h:318-328); value d*sc*q-dmin*m
    // (ggml-quants.c:1471-1488). q∈[0,15], 8 sub-blocks of 32. amax from q min/max (affine).
    for b in buf.chunks_exact(144) {
        let d = f16_to_f32(u16le(b));
        let dmin = f16_to_f32(u16le(&b[2..4]));
        let scales = &b[4..16];
        let qs = &b[16..144];
        for is in 0..8usize {
            let (sc, m) = scale_min_k4(is, scales);
            let dl = d * sc as f32;
            let ml = dmin * m as f32;
            let mut blk = Blk::default();
            let mut qmin = 16i32;
            let mut qmax = 0i32;
            for l in 0..32 {
                let q = qs_nibble(qs, is, l) as i32;
                qmin = qmin.min(q);
                qmax = qmax.max(q);
                blk.add(dl * q as f32 - ml);
            }
            let vlo = (dl * qmin as f32 - ml) as f64;
            let vhi = (dl * qmax as f32 - ml) as f64;
            blk.finish_amax(vlo.abs().max(vhi.abs()));
            sink.block(&blk);
        }
    }
}

pub(crate) fn scan_q5_k(buf: &[u8], sink: &mut RowSink) {
    // block_q5_K: d,dmin | scales[12] | qh[32] | qs[128] (ggml-common.h:332-346);
    // q = nib + 16*bit, bit is of qh[l] (ggml-quants.c:1673-1704). 8 sub-blocks of 32.
    for b in buf.chunks_exact(176) {
        let d = f16_to_f32(u16le(b));
        let dmin = f16_to_f32(u16le(&b[2..4]));
        let scales = &b[4..16];
        let qh = &b[16..48];
        let qs = &b[48..176];
        for is in 0..8usize {
            let (sc, m) = scale_min_k4(is, scales);
            let dl = d * sc as f32;
            let ml = dmin * m as f32;
            let mut blk = Blk::default();
            let mut qmin = 32i32;
            let mut qmax = 0i32;
            for l in 0..32 {
                let q = (qs_nibble(qs, is, l) | ((((qh[l] >> is) as u32) & 1) << 4)) as i32;
                qmin = qmin.min(q);
                qmax = qmax.max(q);
                blk.add(dl * q as f32 - ml);
            }
            let vlo = (dl * qmin as f32 - ml) as f64;
            let vhi = (dl * qmax as f32 - ml) as f64;
            blk.finish_amax(vlo.abs().max(vhi.abs()));
            sink.block(&blk);
        }
    }
}

pub(crate) fn scan_q6_k(buf: &[u8], sink: &mut RowSink) {
    // block_q6_K: ql[128] | qh[64] | scales[16] int8 | d f16 (ggml-common.h:348-358).
    // 16 sub-blocks of 16: v = d*sc[i]*q, q∈[-32,31] (ggml-quants.c:1881-1908).
    // Census-32 block = sub-blocks 2k, 2k+1 (amax = max of the two, sum_sq = sum).
    for b in buf.chunks_exact(210) {
        let ql = &b[0..128];
        let qh = &b[128..192];
        let scq = &b[192..208];
        let d = f16_to_f32(u16le(&b[208..210]));
        let mut blk = Blk::default();
        for i in 0..16usize {
            let n = i / 8;
            let fi = (i % 8) / 2;
            let dl = d * scq[i] as f32;
            let mut sub_abs_max: f64 = 0.0;
            for l in 0..16 {
                let ql_byte = ql[64 * n + 32 * (fi % 2) + l];
                let qlc = if fi < 2 { ql_byte } else { ql_byte >> 4 };
                let base = (qlc & 0xF) as u32;
                let hi = (((qh[32 * n + l] >> (2 * fi)) & 3) << 4) as u32;
                let qi = (base | hi) as i32 - 32;
                let v = dl * qi as f32;
                let a = (v as f64).abs();
                if a > sub_abs_max {
                    sub_abs_max = a;
                }
                blk.add(v);
            }
            blk.finish_amax(sub_abs_max);
			if i % 2 == 1 {
                sink.block(&blk);
                blk = Blk::default();
            }
        }
    }
}

pub(crate) fn scan_iq4_nl(buf: &[u8], sink: &mut RowSink) {
    // block_iq4_nl: d f16 | 16 bytes nibbles; v = d*kvalues[nib] (ggml-quants.c:2653-2667).
    for b in buf.chunks_exact(18) {
        let d = f16_to_f32(u16le(b));
        let mut blk = Blk::default();
        let mut kmax: f64 = 0.0;
        for j in 0..16 {
            let a = b[2 + j] & 0xF;
            let bb = b[2 + j] >> 4;
            let ka = (KVALUES_IQ4NL[a as usize] as f64).abs();
            let kb = (KVALUES_IQ4NL[bb as usize] as f64).abs();
            kmax = kmax.max(ka).max(kb);
            blk.add(d * KVALUES_IQ4NL[a as usize]);
            blk.add(d * KVALUES_IQ4NL[bb as usize]);
        }
        blk.finish_amax((d as f64).abs() * kmax);
        sink.block(&blk);
    }
}

pub(crate) fn scan_iq4_xs(buf: &[u8], sink: &mut RowSink) {
    // block_iq4_xs: d f16 | scales_h u16 | scales_l[4] | qs[128] (ggml-common.h:448-450).
    // ls = (scales_l[ib/2]>>4*(ib%2)&0xF)|((scales_h>>2*ib)&3)<<4; dl=d*(ls-32) (ggml-quants.c:2671-2697).
    for b in buf.chunks_exact(136) {
        let d = f16_to_f32(u16le(b));
        let scales_h = u16::from_le_bytes([b[2], b[3]]);
        let scales_l = &b[4..8];
        let qs = &b[8..136];
        for ib in 0..8usize {
            let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as u32)
                | ((((scales_h >> (2 * ib)) & 3) as u32) << 4);
            let dl = d * ((ls as i32) - 32) as f32;
            let mut blk = Blk::default();
            let mut kmax: f64 = 0.0;
            for j in 0..16 {
                let a = qs[16 * ib + j] & 0xF;
                let bb = qs[16 * ib + j] >> 4;
                let ka = (KVALUES_IQ4NL[a as usize] as f64).abs();
                let kb = (KVALUES_IQ4NL[bb as usize] as f64).abs();
                kmax = kmax.max(ka).max(kb);
                blk.add(dl * KVALUES_IQ4NL[a as usize]);
                blk.add(dl * KVALUES_IQ4NL[bb as usize]);
            }
            blk.finish_amax((dl as f64).abs() * kmax);
            sink.block(&blk);
        }
    }
}

pub(crate) fn scan_q2_k(buf: &[u8], sink: &mut RowSink) {
    // block_q2_K: d,dmin f16 | scales[16] | qs[64] (ggml-common.h:265-272);
    // 8 sub-blocks of 16: sub-scale & sub-min are nibbles of scales[is]; q 2-bit (ggml-quants.c:903-929).
    for b in buf.chunks_exact(84) {
        let d = f16_to_f32(u16le(b));
        let dmin = f16_to_f32(u16le(&b[2..4]));
        let scales = &b[4..20];
        let qs = &b[20..84];
        let mut blk = Blk::default();
        for is in 0..8usize {
            let dl = d * (scales[is] & 0xF) as f32;
            let ml = dmin * (scales[is] >> 4) as f32;
            let qbase = 32 * (is / 8) + 16 * (is % 2);
            let shift = 2 * ((is % 8) / 2);
            let mut qmin = 4i32;
            let mut qmax = 0i32;
            for l in 0..16 {
                let q = ((qs[qbase + l] >> shift) & 3) as i32;
                qmin = qmin.min(q);
                qmax = qmax.max(q);
                blk.add(dl * q as f32 - ml);
            }
            let vlo = (dl * qmin as f32 - ml) as f64;
            let vhi = (dl * qmax as f32 - ml) as f64;
            blk.finish_amax(vlo.abs().max(vhi.abs()));
            if is % 2 == 1 {
                sink.block(&blk);
                blk = Blk::default();
            }
        }
    }
}

/// q3_K scale unpack: 12 bytes -> 16 int8 via the aux shuffle (ggml-quants.c:1263-1273).
fn q3_k_scales(scales12: &[u8]) -> [i8; 16] {
    let mut aux = [0u32; 4];
    for (i, w) in aux.iter_mut().enumerate() {
        *w = u32::from_le_bytes(scales12[4 * i..4 * i + 4].try_into().unwrap());
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
    aux[0] = (aux[0] & 0x0f0f0f0f) | (((tmp >> 0) & 0x03030303) << 4);
    aux[1] = (aux[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4);
    let mut out = [0i8; 16];
    for i in 0..16 {
        out[i] = ((aux[i / 4] >> (8 * (i % 4))) & 0xFF) as i8;
    }
    out
}

pub(crate) fn scan_q3_k(buf: &[u8], sink: &mut RowSink) {
    // block_q3_K: hmask[32] | qs[64] | scales[12] | d f16 (ggml-common.h:274-286);
    // 16 sub-blocks of 16: v = d*(sc-32)*q, q∈[-4,3] (ggml-quants.c:1247-1294).
    for b in buf.chunks_exact(110) {
        let hm = &b[0..32];
        let qs = &b[32..96];
        let scales = q3_k_scales(&b[96..108]);
        let d = f16_to_f32(u16le(&b[108..110]));
        let mut blk = Blk::default();
        for is in 0..16usize {
            let dl = d * ((scales[is] as i32) - 32) as f32;
            let qbase = 32 * (is / 8) + 16 * (is % 2);
            let shift = 2 * ((is % 8) / 2);
            let mbit = 1u8 << (4 * (is / 8) + (is % 8) / 2);
            let mut qmin = 5i32;
            let mut qmax = -5i32;
            for l in 0..16 {
                let q = ((qs[qbase + l] >> shift) & 3) as i32;
                let qv = if (hm[qbase + l] & mbit) != 0 { q } else { q - 4 };
                qmin = qmin.min(qv);
                qmax = qmax.max(qv);
                blk.add(dl * qv as f32);
            }
            blk.finish_amax((dl as f64).abs() * (qmin.abs().max(qmax.abs())) as f64);
            if is % 2 == 1 {
                sink.block(&blk);
                blk = Blk::default();
            }
        }
    }
}

/// Float dtypes: decode the row to f32 and feed the dequantized path (native tail handling).
fn scan_float_row(buf: &[u8], ty: GType, acc: &mut StatAcc) {
    let mut vals: Vec<f32> = Vec::with_capacity(buf.len() / 2 + 1);
    match ty {
        GType::F32 => {
            for b in buf.chunks_exact(4) {
                vals.push(f32::from_le_bytes(b.try_into().unwrap()));
            }
        }
        GType::F16 => {
            for b in buf.chunks_exact(2) {
                vals.push(f16_to_f32(u16le(b)));
            }
        }
        GType::Bf16 => {
            for b in buf.chunks_exact(2) {
                vals.push(bf16_to_f32(u16le(b)));
            }
        }
        _ => unreachable!(),
    }
    acc.feed_row(&vals);
}

/// Scan one tensor's row stream into `acc` (structural for quant, dequantized for float).
fn scan_tensor_rows(f: &mut BufReader<File>, t: &TensorMeta, acc: &mut StatAcc) -> Result<(), String> {
    let row_el = t.ne[0] as usize;
    let (blk_el, blk_by) = t.ty.blk();
    let nblocks_row = row_el / blk_el;
    if nblocks_row == 0 {
        return Err("zero-width row".into());
    }
    let row_bytes = nblocks_row * blk_by;
    let n_rows = t.ne[1] as usize;

    if let GType::F32 | GType::F16 | GType::Bf16 = t.ty {
        let mut raw = vec![0u8; row_bytes];
        for _ in 0..n_rows {
            f.read_exact(&mut raw).map_err(|_| "short row (float)".to_string())?;
            scan_float_row(&raw, t.ty, acc);
        }
        return Ok(());
    }

    let mut buf = vec![0u8; row_bytes];
    let mut sink = RowSink::new(acc);
    for _ in 0..n_rows {
        f.read_exact(&mut buf).map_err(|_| "short row (quant)".to_string())?;
        match t.ty {
            GType::Q8_0 => scan_q8_0(&buf, &mut sink),
            GType::Q8_K => scan_q8_k(&buf, &mut sink),
            GType::Q4_0 => scan_q4_0(&buf, &mut sink),
            GType::Q4_1 => scan_q4_1(&buf, &mut sink),
            GType::Q5_0 => scan_q5_0(&buf, &mut sink),
            GType::Q5_1 => scan_q5_1(&buf, &mut sink),
            GType::Q1_0 => scan_q1_0(&buf, &mut sink),
            GType::Q4_K => scan_q4_k(&buf, &mut sink),
            GType::Q5_K => scan_q5_k(&buf, &mut sink),
            GType::Q6_K => scan_q6_k(&buf, &mut sink),
            GType::Q2_K => scan_q2_k(&buf, &mut sink),
            GType::Q3_K => scan_q3_k(&buf, &mut sink),
            GType::Iq4Nl => scan_iq4_nl(&buf, &mut sink),
            GType::Iq4Xs => scan_iq4_xs(&buf, &mut sink),
            _ => unreachable!(),
        }
        sink.end_row();
    }
    Ok(())
}

/// Drive the whole GGUF artifact: header parse + per-tensor census.
pub fn scan_gguf(path: &str, ctx: &GgufCtx) -> Result<ScanOut, String> {
    let f = File::open(path).map_err(|e| e.to_string())?;
    let mut br = BufReader::with_capacity(1 << 20, f);
    let mut per_family: BTreeMap<&'static str, StatAcc> = BTreeMap::new();
    let mut total = StatAcc::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    let mut metered = 0usize;
    let mut type_counts: BTreeMap<&'static str, usize> = BTreeMap::new();

    for t in &ctx.tensors {
        let fam = match gguf_family_of(&t.name) {
            Some(x) => x,
            None => continue,
        };
        if t.n_dims != 2 || t.ne[0] == 0 || t.ne[1] == 0 {
            skipped.push((t.name.clone(), format!("rank {} skipped", t.n_dims)));
            continue;
        }
        if !t.ty.supported() {
            // IQ1/2/3 latent-grid types need the codebook tables + lattice index math to rebuild
            // values; not representable structurally here, so they are skipped, never faked.
            let code = match t.ty {
                GType::Unsupported(c) => c,
                _ => 0,
            };
            skipped.push((
                t.name.clone(),
                format!("unsupported quant type (code {}): latent grid, no structural amax", code),
            ));
            continue;
        }
        let abs_off = ctx.data_start + t.offset;
        br.seek(SeekFrom::Start(abs_off)).map_err(|e| e.to_string())?;
        let mut acc = StatAcc::new();
        scan_tensor_rows(&mut br, t, &mut acc).map_err(|e| format!("{}: {}", t.name, e))?;
        acc.close_tensor();
        metered += 1;
        *type_counts.entry(t.ty.name()).or_insert(0) += 1;
        total.merge(&acc);
        per_family.entry(fam).or_insert_with(StatAcc::new).merge(&acc);
    }

    Ok(ScanOut {
        per_family,
        total,
        skipped,
        metered_tensors: metered,
        type_counts,
    })
}
