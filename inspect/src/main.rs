//! theseus-inspect — static checkpoint diagnostics, zero dependencies.
//!
//! Question it answers: *what can still safely be done to this checkpoint?* Not by running
//! surgery, but by reading the bytes that surgery will act on. M1 measured that a
//! function-preserving RMSNorm-diagonal gauge (bit-identical logits) makes Q8_0 diverge by
//! 10.7 nats and destroys bounded-LoRA capture (0.973 -> 0.156); every number below is computable
//! from the artifact alone, in milliseconds, and was pre-registered before those results existed.
//!
//! It parses the safetensors container directly (8-byte u64 header length, JSON header, tensor
//! data after it) so it needs no crate, no torch, and never loads the model.
//!
//! Metrics per tensor, and aggregated per weight family:
//!   * `dyn_range_log10`   log10(max|w| / min|w|>0) — how much dynamic range the family spans
//!   * `below_f16_normal`  count of nonzero |w| under 2^-14 (6.1035e-5): those weights are
//!                         destroyed by the f16 export every quantize script uses
//!   * `q4_block_mse`      predicted relative MSE of 4-bit block-max-abs symmetric quantization
//!                         with 32-element contiguous blocks (the unit llama.cpp k-quants use):
//!                         sum_blocks amax^2 * n / (12 * 7^2 * sum w^2)
//!   * `row_energy_imbal`  max/min row L2 energy — how unequal the optimizer's per-coordinate
//!                         geometry is; the LoRA-collapse candidate feature
//!   * `amax_over_rms`     global amax / rms: outlier dominance (what clipping and scales chase)

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::process::exit;

const F16_NORMAL_MIN: f32 = 6.103515625e-5; // 2^-14
const BLOCK: usize = 32;
const QBITS: f64 = 7.0; // 2^(4-1) - 1 for symmetric 4-bit

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dtype {
    F32,
    F16,
    Bf16,
    Other,
}

impl Dtype {
    fn parse(s: &str) -> Dtype {
        match s {
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::Bf16,
            _ => Dtype::Other,
        }
    }
    fn size(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::Bf16 => 2,
            Dtype::Other => 0,
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    let v = if exp == 0 {
        mant * 2f32.powi(-24) // subnormal, exact in f32
    } else if exp == 31 {
        return if mant != 0.0 { f32::NAN } else { f32::INFINITY };
    } else {
        (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
    };
    if sign == 1 {
        -v
    } else {
        v
    }
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Minimal JSON reader for the safetensors header: we only need string keys mapped to
/// {dtype: str, shape: [u64], data_offsets: [u64, u64]}.
mod header {
    pub struct Toks<'a> {
        s: &'a [u8],
        pub i: usize,
    }

    impl<'a> Toks<'a> {
        pub fn new(s: &'a [u8]) -> Toks<'a> {
            Toks { s, i: 0 }
        }
        pub fn ws(&mut self) {
            while self.i < self.s.len() && (self.s[self.i] as char).is_whitespace() {
                self.i += 1;
            }
        }
        pub fn peek(&mut self) -> Option<u8> {
            self.ws();
            self.s.get(self.i).copied()
        }
        pub fn eat(&mut self, c: u8) -> Result<(), String> {
            if self.peek() == Some(c) {
                self.i += 1;
                Ok(())
            } else {
                Err(format!("expected '{}' at byte {}", c as char, self.i))
            }
        }
        pub fn string(&mut self) -> Result<String, String> {
            self.eat(b'"')?;
            let start = self.i;
            while self.i < self.s.len() {
                match self.s[self.i] {
                    b'"' => {
                        let out = String::from_utf8_lossy(&self.s[start..self.i]).to_string();
                        self.i += 1;
                        return Ok(out);
                    }
                    b'\\' => self.i += 2,
                    _ => self.i += 1,
                }
            }
            Err("unterminated string".into())
        }
        pub fn number(&mut self) -> Result<u64, String> {
            self.ws();
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
            if start == self.i {
                return Err(format!("expected number at byte {}", self.i));
            }
            std::str::from_utf8(&self.s[start..self.i])
                .map_err(|e| e.to_string())?
                .parse::<u64>()
                .map_err(|e| e.to_string())
        }
        pub fn null(&mut self) -> Result<(), String> {
            self.ws();
            if self.s[self.i..].starts_with(b"null") {
                self.i += 4;
                Ok(())
            } else {
                Err("expected null".into())
            }
        }
        pub fn skip_value(&mut self) -> Result<(), String> {
            match self.peek() {
                Some(b'"') => {
                    self.string()?;
                    Ok(())
                }
                Some(b'[') => {
                    self.eat(b'[')?;
                    if self.peek() == Some(b']') {
                        self.i += 1;
                        return Ok(());
                    }
                    loop {
                        self.skip_value()?;
                        match self.peek() {
                            Some(b',') => {
                                self.i += 1;
                            }
                            Some(b']') => {
                                self.i += 1;
                                return Ok(());
                            }
                            _ => return Err("bad array".into()),
                        }
                    }
                }
                Some(b'{') => {
                    self.eat(b'{')?;
                    if self.peek() == Some(b'}') {
                        self.i += 1;
                        return Ok(());
                    }
                    loop {
                        self.string()?;
                        self.eat(b':')?;
                        self.skip_value()?;
                        match self.peek() {
                            Some(b',') => {
                                self.i += 1;
                            }
                            Some(b'}') => {
                                self.i += 1;
                                return Ok(());
                            }
                            _ => return Err("bad object".into()),
                        }
                    }
                }
                Some(b'n') => self.null(),
                Some(_) => {
                    self.number()?;
                    Ok(())
                }
                None => Err("unexpected end".into()),
            }
        }
        pub fn u64_array(&mut self) -> Result<Vec<u64>, String> {
            self.eat(b'[')?;
            let mut out = Vec::new();
            if self.peek() == Some(b']') {
                self.i += 1;
                return Ok(out);
            }
            loop {
                out.push(self.number()?);
                match self.peek() {
                    Some(b',') => {
                        self.i += 1;
                    }
                    Some(b']') => {
                        self.i += 1;
                        return Ok(out);
                    }
                    _ => return Err("bad number array".into()),
                }
            }
        }
        /// One tensor entry: { "dtype": ..., "shape": [...], "data_offsets": [a,b] }
        pub fn tensor_entry(&mut self) -> Result<(String, Vec<u64>, u64, u64), String> {
            self.eat(b'{')?;
            let mut dtype = String::new();
            let mut shape: Vec<u64> = Vec::new();
            let mut off = (0u64, 0u64);
            if self.peek() == Some(b'}') {
                self.i += 1;
                return Err("empty tensor entry".into());
            }
            loop {
                let key = self.string()?;
                self.eat(b':')?;
                match key.as_str() {
                    "dtype" => dtype = self.string()?,
                    "shape" => shape = self.u64_array()?,
                    "data_offsets" => {
                        let v = self.u64_array()?;
                        if v.len() != 2 {
                            return Err("data_offsets must have 2 entries".into());
                        }
                        off = (v[0], v[1]);
                    }
                    _ => self.skip_value()?,
                }
                match self.peek() {
                    Some(b',') => {
                        self.i += 1;
                    }
                    Some(b'}') => {
                        self.i += 1;
                        break;
                    }
                    _ => return Err("bad tensor entry".into()),
                }
            }
            Ok((dtype, shape, off.0, off.1))
        }
    }
}

#[derive(Default, Clone)]
struct Acc {
    n: u64,
    sum_sq: f64,
    amax: f64,
    cur_block_amax_sq: f64,
    cur_sum_sq: f64,
    amin_nz: f64,
    below_f16: u64,
    block_amax_sq: f64, // sum over 32-blocks of amax^2 * BLOCK
    blocks: u64,
    row_max_e: f64,
    row_min_e: f64,
    rows: u64,
    /// per-tensor mean of the ratio, which is the convention the M1 predictions were registered
    /// under (`canonicalize.quant_condition` averages per-tensor numbers, not pooled ones)
    tensor_ratio_sum: f64,
    tensor_count: u64,
}

impl Acc {
    fn new() -> Acc {
        Acc {
            amin_nz: f64::INFINITY,
            row_min_e: f64::INFINITY,
            ..Default::default()
        }
    }
    /// Feed one row (contiguous) of values.
    fn feed_row(&mut self, row: &[f32]) {
        let mut row_sq = 0f64;
        let mut chunks = row.len();
        let mut base = 0usize;
        while chunks > 0 {
            let k = if chunks < BLOCK { chunks } else { BLOCK };
            let mut bmax = 0f64;
            for &v in &row[base..base + k] {
                let a = v.abs() as f64;
                self.n += 1;
                row_sq += a * a;
                self.sum_sq += a * a;
                self.cur_sum_sq += a * a;
                if a > self.amax {
                    self.amax = a;
                }
                if a > 0.0 && a < self.amin_nz {
                    self.amin_nz = a;
                }
                if a > 0.0 && a < F16_NORMAL_MIN as f64 {
                    self.below_f16 += 1;
                }
                if a > bmax {
                    bmax = a;
                }
            }
            if k == BLOCK {
                self.block_amax_sq += bmax * bmax * BLOCK as f64;
                self.cur_block_amax_sq += bmax * bmax * BLOCK as f64;
                self.blocks += 1;
            }
            base += k;
            chunks -= k;
        }
        if row_sq > 0.0 {
            self.rows += 1;
            if row_sq > self.row_max_e {
                self.row_max_e = row_sq;
            }
            if row_sq < self.row_min_e {
                self.row_min_e = row_sq;
            }
        }
    }
    fn close_tensor(&mut self) {
        if self.cur_sum_sq > 0.0 && self.cur_block_amax_sq > 0.0 {
            self.tensor_ratio_sum +=
                self.cur_block_amax_sq / (12.0 * QBITS * QBITS * self.cur_sum_sq);
            self.tensor_count += 1;
        }
        self.cur_block_amax_sq = 0.0;
        self.cur_sum_sq = 0.0;
    }
    fn merge(&mut self, o: &Acc) {
        self.tensor_ratio_sum += o.tensor_ratio_sum;
        self.tensor_count += o.tensor_count;
        self.n += o.n;
        self.sum_sq += o.sum_sq;
        self.amax = self.amax.max(o.amax);
        self.amin_nz = self.amin_nz.min(o.amin_nz);
        self.below_f16 += o.below_f16;
        self.block_amax_sq += o.block_amax_sq;
        self.blocks += o.blocks;
        self.row_max_e = self.row_max_e.max(o.row_max_e);
        self.row_min_e = self.row_min_e.min(o.row_min_e);
        self.rows += o.rows;
    }
    fn report(&self) -> BTreeMap<&'static str, f64> {
        let mut m = BTreeMap::new();
        let rms = if self.n > 0 {
            (self.sum_sq / self.n as f64).sqrt()
        } else {
            0.0
        };
        m.insert(
            "dyn_range_log10",
            if self.amin_nz.is_finite() && self.amin_nz > 0.0 && self.amax > 0.0 {
                (self.amax / self.amin_nz).log10()
            } else {
                0.0
            },
        );
        m.insert("q4_block_mse_pooled", if self.sum_sq > 0.0 { self.block_amax_sq / (12.0 * QBITS * QBITS * self.sum_sq) } else { 0.0 });
        m.insert("q4_block_mse", if self.tensor_count > 0 { self.tensor_ratio_sum / self.tensor_count as f64 } else { 0.0 });
        m.insert(
            "row_energy_imbalance",
            if self.rows > 0 && self.row_min_e.is_finite() && self.row_min_e > 0.0 {
                self.row_max_e / self.row_min_e
            } else {
                0.0
            },
        );
        m.insert("amax_over_rms", if rms > 0.0 { self.amax / rms } else { 0.0 });
        m.insert("below_f16_normal", self.below_f16 as f64);
        m.insert("frac_below_f16_normal", if self.n > 0 { self.below_f16 as f64 / self.n as f64 } else { 0.0 });
        m.insert("weights", self.n as f64);
        m
    }
}

fn family_of(name: &str) -> Option<&'static str> {
    for f in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] {
        if name.contains(f) {
            return Some(match f {
                "q_proj" => "q_proj",
                "k_proj" => "k_proj",
                "v_proj" => "v_proj",
                "o_proj" => "o_proj",
                "gate_proj" => "gate_proj",
                "up_proj" => "up_proj",
                _ => "down_proj",
            });
        }
    }
    None
}

fn usage() -> ! {
    eprintln!("usage: theseus-inspect <model.safetensors> [--json out.json] [--fail-above FRAC]");
    eprintln!();
    eprintln!("  reads a safetensors artifact and prints per-family static conditioning:");
    eprintln!("    q4_block_mse         predicted 4-bit block-max-abs relative MSE (32-blocks)");
    eprintln!("    dyn_range_log10      log10(amax / min nonzero |w|)");
    eprintln!("    row_energy_imbalance max/min row L2 energy (per-coordinate optimizer geometry)");
    eprintln!("    frac_below_f16_normal share of weights the f16 GGUF export cannot represent");
    eprintln!("    amax_over_rms        outlier dominance");
    eprintln!("  exit 1 when --fail-above is given and any family exceeds that fraction.");
    exit(2)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut path: Option<String> = None;
    let mut json_out: Option<String> = None;
    let mut fail_above: Option<f64> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => {
                json_out = args.get(i + 1).cloned();
                i += 2;
            }
            "--fail-above" => {
                fail_above = args
                    .get(i + 1)
                    .and_then(|s| s.parse::<f64>().ok());
                i += 2;
            }
            "-h" | "--help" => usage(),
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
    let mut f = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: cannot open {}: {}", path, e);
            exit(1);
        }
    };
    let mut len_buf = [0u8; 8];
    if f.read_exact(&mut len_buf).is_err() {
        eprintln!("error: {}: not a safetensors file (no header length)", path);
        exit(1);
    }
    let header_len = u64::from_le_bytes(len_buf) as usize;
    if header_len == 0 || header_len > (1 << 28) {
        eprintln!("error: implausible header length {}", header_len);
        exit(1);
    }
    let mut hb = vec![0u8; header_len];
    if f.read_exact(&mut hb).is_err() {
        eprintln!("error: truncated header");
        exit(1);
    }
    let data_start = 8u64 + header_len as u64;

    // Parse entries.
    let mut p = header::Toks::new(&hb);
    p.eat(b'{').unwrap_or_else(|e| fatal(e.as_str()));
    let mut entries: Vec<(String, Dtype, Vec<u64>, u64, u64)> = Vec::new();
    if p.peek() == Some(b'}') {
        p.i += 1;
    } else {
        loop {
            let key = p.string().unwrap_or_else(|e| fatal(e.as_str()));
            p.eat(b':').unwrap_or_else(|e| fatal(e.as_str()));
            if key == "__metadata__" {
                p.skip_value().unwrap_or_else(|e| fatal(e.as_str()));
            } else {
                let (dtype, shape, a, b) = p.tensor_entry().unwrap_or_else(|e| fatal(e.as_str()));
                entries.push((key, Dtype::parse(&dtype), shape, a, b));
            }
            match p.peek() {
                Some(b',') => p.i += 1,
                Some(b'}') => {
                    p.i += 1;
                    break;
                }
                _ => fatal("bad top-level object"),
            }
            if p.i >= hb.len() {
                break;
            }
        }
    }

    let mut per_family: BTreeMap<&'static str, Acc> = BTreeMap::new();
    let mut total = Acc::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    let mut rows_seen = 0usize;

    for (name, dt, shape, a, b) in &entries {
        let fam = match family_of(name) {
            Some(x) => x,
            None => continue,
        };
        if *dt == Dtype::Other || shape.is_empty() {
            skipped.push((name.clone(), format!("dtype/{:?}", dt)));
            continue;
        }
        let esz = dt.size();
        let n_el: u64 = shape.iter().product();
        if b - a != n_el * esz as u64 {
            skipped.push((name.clone(), "size mismatch".into()));
            continue;
        }
        // Row length = last dimension for 2-D; 1-D norms are not surgery targets we meter here.
        let cols = *shape.last().unwrap() as usize;
        if shape.len() != 2 || cols == 0 {
            skipped.push((name.clone(), format!("rank {} skipped", shape.len())));
            continue;
        }
        let row_bytes = cols * esz;
        let mut acc = Acc::new();
        let mut buf = vec![0u8; row_bytes];
        let mut vals: Vec<f32> = vec![0.0; cols];
        for r in 0..shape[0] as usize {
            f.seek(SeekFrom::Start(data_start + a + (r * row_bytes) as u64))
                .unwrap_or_else(|e| fatal(e.to_string().as_str()));
            if f.read_exact(&mut buf).is_err() {
                fatal("short read");
            }
            match dt {
                Dtype::F32 => {
                    for k in 0..cols {
                        vals[k] = f32::from_le_bytes([
                            buf[4 * k],
                            buf[4 * k + 1],
                            buf[4 * k + 2],
                            buf[4 * k + 3],
                        ]);
                    }
                }
                Dtype::Bf16 => {
                    for k in 0..cols {
                        vals[k] = bf16_to_f32(u16::from_le_bytes([buf[2 * k], buf[2 * k + 1]]));
                    }
                }
                Dtype::F16 => {
                    for k in 0..cols {
                        vals[k] = f16_to_f32(u16::from_le_bytes([buf[2 * k], buf[2 * k + 1]]));
                    }
                }
                Dtype::Other => unreachable!(),
            }
            acc.feed_row(&vals);
        }
        acc.close_tensor();
        rows_seen += 1;
        total.merge(&acc);
        per_family.entry(fam).or_insert_with(Acc::new).merge(&acc);
    }

    println!(
        "theseus-inspect  {}  ({} 2-D weight tensors metered, {} skipped)",
        path,
        rows_seen,
        skipped.len()
    );
    println!(
        "{:<12} {:>9} {:>10} {:>12} {:>10} {:>11} {:>12}",
        "family", "q4_mse", "dyn_rng", "row_imbal", "amax/rms", "frac<f16n", "weights"
    );
    let mut json = String::from("{\n  \"path\": \"");
    json.push_str(&path.replace('\\', "\\\\").replace('"', "\\\""));
    json.push_str("\",\n  \"families\": {\n");
    let mut worst_frac = 0f64;
    let mut first = true;
    for (fam, acc) in &per_family {
        let r = acc.report();
        let mf = r["frac_below_f16_normal"];
        worst_frac = worst_frac.max(mf);
        println!(
            "{:<12} {:>9.5} {:>10.3} {:>12.1} {:>10.1} {:>11.6} {:>12}",
            fam,
            r["q4_block_mse"],
            r["dyn_range_log10"],
            r["row_energy_imbalance"],
            r["amax_over_rms"],
            mf,
            r["weights"] as u64
        );
        if !first {
            json.push_str(",\n");
        }
        first = false;
        json.push_str(&format!("    \"{}\": {{", fam));
        let mut inner = true;
        for (k, v) in &r {
            if !inner {
                json.push_str(", ");
            }
            inner = false;
            json.push_str(&format!("\"{}\": {:.8}", k, v));
        }
        json.push('}');
    }
    let tr = total.report();
    json.push_str("\n  },\n  \"total\": {");
    let mut inner = true;
    for (k, v) in &tr {
        if !inner {
            json.push_str(", ");
        }
        inner = false;
        json.push_str(&format!("\"{}\": {:.8}", k, v));
    }
    json.push_str("},\n  \"skipped\": [");
    for (i, (n, why)) in skipped.iter().take(12).enumerate() {
        if i > 0 {
            json.push_str(", ");
        }
        json.push_str(&format!("[\"{}\", \"{}\"]", n.replace('"', "'"), why));
    }
    json.push_str("]\n}\n");
    println!(
        "{:<12} {:>9.5} {:>10.3} {:>12.1} {:>10.1} {:>11.6} {:>12}",
        "TOTAL",
        tr["q4_block_mse"],
        tr["dyn_range_log10"],
        tr["row_energy_imbalance"],
        tr["amax_over_rms"],
        tr["frac_below_f16_normal"],
        tr["weights"] as u64
    );
    if let Some(o) = json_out {
        if let Err(e) = std::fs::write(&o, json) {
            eprintln!("error: writing {}: {}", o, e);
            exit(1);
        }
        eprintln!("wrote {}", o);
    }
    if let Some(thr) = fail_above {
        if worst_frac > thr {
            eprintln!(
                "FAIL: a family has {:.4} of its weights below the f16 normal range (> {:.4})",
                worst_frac, thr
            );
            exit(1);
        }
        eprintln!("OK: no family exceeds the f16 exportability threshold");
    }
}

fn fatal(msg: &str) -> ! {
    eprintln!("error: header parse: {}", msg);
    exit(1)
}
