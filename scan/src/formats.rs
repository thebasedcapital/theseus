//! formats.rs — container sniffing + safetensors reader + PEFT/LoRA adapter detection (owner: ScanFormats).
//!
//! Sniffs by magic bytes first, extension second:
//!   * "GGUF" magic bytes           -> GGUF (parsed by gguf.rs)
//!   * 8-byte LE header len + '{'   -> safetensors (parsed here); keys are then checked for
//!                                     lora_A/lora_B markers -> a PEFT/LoRA ADAPTER, which is
//!                                     reported with its own metric set and never scanned as a model.
//!   * ".bin" torch pickle          -> reported UNAVAILABLE (needs torch; bytes cannot tell us).
//! The safetensors reader is a self-contained port of inspect/src/main.rs's minimal JSON header
//! tokenizer (no crate, no whole-model load; rows are streamed with BufReader).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::stats::{bf16_to_f32, f16_to_f32, StatAcc};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Container {
    Safetensors,
    Gguf,
    PyTorchBin,
    NumpyPair,
    Unknown,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArtifactKind {
    FullModel,
    Adapter,
    Unsupported,
}

pub struct Sniff {
    pub container: Container,
    pub kind: ArtifactKind,
    pub reason: String,
}

pub const ADAPTER_VERDICT: &str = "ADAPTER: static risk flags not defined for adapters yet";

/// Sniff a file without parsing the whole thing. Never symmetric-attacks: a truncated header
/// yields Unsupported, never a wrong container.
pub fn sniff(path: &str) -> Sniff {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            return Sniff {
                container: Container::Unknown,
                kind: ArtifactKind::Unsupported,
                reason: format!("cannot open {}: {}", path, e),
            }
        }
    };
    // Read as much of the 16-byte head as exists; a short file is still classified by what it
    // has plus its extension, never by padding and never by a whole-file guess.
    let mut head = [0u8; 16];
    let mut n = 0usize;
    while n < head.len() {
        match f.read(&mut head[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(_) => break,
        }
    }
    if n == 0 {
        return Sniff {
            container: Container::Unknown,
            kind: ArtifactKind::Unsupported,
            reason: format!("{}: empty file", path),
        };
    }
    // 1) GGUF magic "GGUF" (needs 4 bytes).
    if n >= 4 && &head[0..4] == b"GGUF" {
        return Sniff {
            container: Container::Gguf,
            kind: ArtifactKind::FullModel,
            reason: "GGUF magic".into(),
        };
    }
    // 2) safetensors: 8-byte LE header length, then a JSON '{' where the header begins.
    if n >= 8 {
        let hlen = u64::from_le_bytes(head[0..8].try_into().unwrap());
        if hlen > 0 && hlen <= (1 << 24) {
            let mut hb = [0u8; 1];
            if f.seek(SeekFrom::Start(8)).is_ok() && f.read_exact(&mut hb).is_ok() && hb[0] == b'{'
            {
                return Sniff {
                    container: Container::Safetensors,
                    kind: ArtifactKind::FullModel,
                    reason: "safetensors magic (8-byte LE header + JSON)".into(),
                };
            }
        }
    }
    // 3) extension hints (only reached when magic is inconclusive).
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".bin") {
        return Sniff {
            container: Container::PyTorchBin,
            kind: ArtifactKind::Unsupported,
            reason:
                ".bin is a torch pickle; requires torch to read, static bytes insufficient (I8)"
                    .into(),
        };
    }
    if lower.ends_with(".npy") || lower.ends_with(".npz") {
        return Sniff {
            container: Container::NumpyPair,
            kind: ArtifactKind::Unsupported,
            reason: "npy/npz: MLX weights are a directory of these; a single-file scan cannot read the set".into(),
        };
    }
    Sniff {
        container: Container::Unknown,
        kind: ArtifactKind::Unsupported,
        reason: format!(
            "{}: unrecognized container (not GGUF, not safetensors, not .bin)",
            path
        ),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    Other,
}
impl Dtype {
    pub fn parse(s: &str) -> Dtype {
        match s {
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::Bf16,
            _ => Dtype::Other,
        }
    }
    pub fn size(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::Bf16 => 2,
            Dtype::Other => 0,
        }
    }
}

/// Adapter metrics — the different metric set for PEFT/LoRA artifacts (rank, targets, count,
/// dtype); alpha is genuinely absent from safetensors bytes for HF PEFT, so it stays null/I8.
#[derive(Default)]
pub struct AdapterInfo {
    pub has_lora_a: bool,
    pub has_lora_b: bool,
    pub rank: Option<u64>, // mode over lora_A shape[0] (uniform in PEFT)
    pub ranks: BTreeMap<u64, u64>, // rank -> how many lora_A tensors carry it
    pub target_modules: Vec<String>,
    pub tensor_count: usize,
    pub pair_count: usize,
    pub dtype: Option<String>,
    pub alpha: Option<f64>,
    pub alpha_reason: Option<&'static str>,
}

/// A parsed safetensors header entry.
pub struct TensorEntry {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub offs: (u64, u64),
}

pub struct SafetensorsHeader {
    pub entries: Vec<TensorEntry>,
    pub adapter: AdapterInfo,
    pub is_adapter: bool,
    pub header_len: u64,
}

/// Minimal JSON reader for the safetensors header (port of inspect/src/main.rs mod header,
/// extended to scan *every* top-level key for lora markers instead of discarding unknowns).
mod js {
    pub struct Toks<'a> {
        s: &'a [u8],
        pub i: usize,
    }
    impl<'a> Toks<'a> {
        pub fn new(s: &'a [u8]) -> Toks<'a> {
            Toks { s, i: 0 }
        }
        fn ws(&mut self) {
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
        fn number(&mut self) -> Result<u64, String> {
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
                            Some(b',') => self.i += 1,
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
                            Some(b',') => self.i += 1,
                            Some(b'}') => {
                                self.i += 1;
                                return Ok(());
                            }
                            _ => return Err("bad object".into()),
                        }
                    }
                }
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
                    Some(b',') => self.i += 1,
                    Some(b']') => {
                        self.i += 1;
                        return Ok(out);
                    }
                    _ => return Err("bad number array".into()),
                }
            }
        }
        /// { "dtype": ..., "shape": [...], "data_offsets": [a,b] }
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
                    Some(b',') => self.i += 1,
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

fn is_lora_a(name: &str) -> bool {
    name.contains("lora_A") || name.contains("lora.A") || name.ends_with("lora_A.weight")
}
fn is_lora_b(name: &str) -> bool {
    name.contains("lora_B") || name.contains("lora.B")
}
fn is_adapter_key(name: &str) -> bool {
    is_lora_a(name) || is_lora_b(name) || name.contains("lora_embedding")
}

/// Extract the module base name immediately before ".lora_A"/".lora_B", e.g.
/// "...self_attn.q_proj.lora_A.weight" -> "q_proj".
fn module_of(name: &str) -> String {
    if let Some(idx) = name.find(".lora_") {
        let pre = &name[..idx];
        if let Some(dot) = pre.rfind('.') {
            return pre[dot + 1..].to_string();
        }
        return pre.to_string();
    }
    if let Some(idx) = name.find(".lora.") {
        let pre = &name[..idx];
        return pre.rsplit('.').next().unwrap_or("").to_string();
    }
    "(unknown)".to_string()
}

/// Classify exact dot-separated tensor components. Expert names follow the audited HF converter
/// layouts (conversion/deepseek.py:386-411; conversion/qwen.py:110-139); fused MoE tensors are
/// reported as unavailable rather than guessed (qwen2_moe/modeling_qwen2_moe.py:287-288).
pub fn family_of(name: &str) -> Option<&'static str> {
    let parts: Vec<&str> = name.split('.').collect();
    if let Some(i) = parts.iter().position(|p| *p == "experts") {
        let next = parts.get(i + 1).copied().unwrap_or("");
        let stem = if next.parse::<u32>().is_ok() {
            parts.get(i + 2).copied().unwrap_or("")
        } else {
            next
        };
        if next.parse::<u32>().is_ok() {
            return match stem {
                "gate_proj" | "w1" => Some("expert_gate"),
                "up_proj" | "w3" => Some("expert_up"),
                "down_proj" | "w2" => Some("expert_down"),
                "gate_up_proj" => Some("__unavailable_expert_fused"),
                _ => None,
            };
        }
        if matches!(stem, "gate_up_proj" | "down_proj") {
            return Some("__unavailable_expert_fused");
        }
        return None;
    }
    if parts.iter().any(|p| matches!(*p, "qkv_proj" | "qkv")) {
        return Some("__unavailable_dense_fused");
    }
    [
        ("q_proj", "q_proj"),
        ("k_proj", "k_proj"),
        ("v_proj", "v_proj"),
        ("o_proj", "o_proj"),
        ("gate_proj", "gate_proj"),
        ("up_proj", "up_proj"),
        ("down_proj", "down_proj"),
    ]
    .iter()
    .find_map(|(needle, family)| parts.iter().any(|p| *p == *needle).then_some(*family))
}

/// Parse the safetensors JSON header (all keys), returning entries and adapter info.
pub fn parse_header(path: &str) -> Result<SafetensorsHeader, String> {
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let mut lb = [0u8; 8];
    f.read_exact(&mut lb)
        .map_err(|_| "no 8-byte header length".to_string())?;
    let hlen = u64::from_le_bytes(lb) as usize;
    if hlen == 0 || hlen > (1 << 28) {
        return Err(format!("implausible header length {}", hlen));
    }
    let mut hb = vec![0u8; hlen];
    f.read_exact(&mut hb)
        .map_err(|_| "truncated header".to_string())?;

    let mut p = js::Toks::new(&hb);
    p.eat(b'{').map_err(|e| format!("header parse: {}", e))?;
    let mut entries: Vec<TensorEntry> = Vec::new();
    let mut adapter = AdapterInfo::default();
    if p.peek() == Some(b'}') {
        p.i += 1;
    } else {
        loop {
            let key = p.string().map_err(|e| format!("header parse: {}", e))?;
            p.eat(b':').map_err(|e| format!("header parse: {}", e))?;
            if key == "__metadata__" {
                p.skip_value().map_err(|e| format!("header parse: {}", e))?;
            } else {
                let (dtype, shape, a, b) = p
                    .tensor_entry()
                    .map_err(|e| format!("header parse: {}", e))?;
                if is_lora_a(&key) {
                    adapter.has_lora_a = true;
                    adapter.tensor_count += 1;
                    if let Some(&r) = shape.first() {
                        *adapter.ranks.entry(r).or_insert(0) += 1;
                    }
                    if adapter.dtype.is_none() {
                        adapter.dtype = Some(dtype.clone());
                    }
                    let m = module_of(&key);
                    if !adapter.target_modules.contains(&m) {
                        adapter.target_modules.push(m);
                    }
                } else if is_lora_b(&key) {
                    adapter.has_lora_b = true;
                    adapter.tensor_count += 1;
                    let m = module_of(&key);
                    if !adapter.target_modules.contains(&m) {
                        adapter.target_modules.push(m);
                    }
                }
                entries.push(TensorEntry {
                    name: key,
                    dtype: Dtype::parse(&dtype),
                    shape,
                    offs: (a, b),
                });
            }
            match p.peek() {
                Some(b',') => p.i += 1,
                Some(b'}') => {
                    p.i += 1;
                    break;
                }
                _ => return Err("bad top-level object".into()),
            }
            if p.i >= hb.len() {
                break;
            }
        }
    }
    let is_adapter = adapter.has_lora_a || adapter.has_lora_b;
    if is_adapter && adapter.has_lora_a {
        // PEFT uses a uniform rank; report the mode (and keep the full histogram).
        adapter.rank = adapter
            .ranks
            .iter()
            .max_by_key(|(_, &c)| c)
            .map(|(r, _)| *r);
        adapter.pair_count = entries
            .iter()
            .filter(|e| is_lora_a(&e.name) || is_lora_b(&e.name))
            .count()
            / 2;
        adapter.alpha = None; // genuine absence: alpha lives in adapter_config.json
        adapter.alpha_reason = Some(
            "safetensors bytes do not carry lora_alpha (HF PEFT stores it in adapter_config.json)",
        );
    } else if is_adapter {
        adapter.alpha = None;
        adapter.alpha_reason = Some("lora_B only; no lora_A to infer rank from");
    }
    Ok(SafetensorsHeader {
        entries,
        adapter,
        is_adapter,
        header_len: hlen as u64,
    })
}

/// Scan a safetensors full model: census per family, exactly inspect/src/main.rs's algorithm,
/// streaming row by row (never materializing a tensor).
pub struct ScanOut {
    pub per_family: BTreeMap<&'static str, StatAcc>,
    pub total: StatAcc,
    pub skipped: Vec<(String, String)>,
    pub metered_tensors: usize,
    pub type_counts: BTreeMap<&'static str, usize>,
}

pub fn scan_safetensors(path: &str, hdr: &SafetensorsHeader) -> Result<ScanOut, String> {
    let data_start = 8u64 + hdr.header_len;
    let f = File::open(path).map_err(|e| e.to_string())?;
    let mut per_family: BTreeMap<&'static str, StatAcc> = BTreeMap::new();
    let mut total = StatAcc::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    let mut metered = 0usize;
    let mut br = BufReader::with_capacity(1 << 20, f);
    for e in &hdr.entries {
        let expert_path = e.name.split('.').any(|p| p == "experts");
        let fam = match family_of(&e.name) {
            Some("__unavailable_expert_fused") => {
                skipped.push((e.name.clone(), format!("UNAVAILABLE: fused expert tensor rank {} cannot be split/deaggregated from header shape alone", e.shape.len())));
                continue;
            }
            Some(x) => x,
            None if expert_path => {
                skipped.push((
                    e.name.clone(),
                    format!(
                        "UNAVAILABLE: unrecognized expert tensor layout rank {}",
                        e.shape.len()
                    ),
                ));
                continue;
            }
            None => continue,
        };
        if e.dtype == Dtype::Other || e.shape.is_empty() {
            skipped.push((e.name.clone(), format!("dtype/{:?}", e.dtype)));
            continue;
        }
        let esz = e.dtype.size();
        let n_el: u64 = e.shape.iter().product();
        if e.offs.1 - e.offs.0 != n_el * esz as u64 {
            skipped.push((e.name.clone(), "size mismatch".into()));
            continue;
        }
        let cols = *e.shape.last().unwrap() as usize;
        if e.shape.len() != 2 || cols == 0 {
            skipped.push((e.name.clone(), format!("rank {} skipped", e.shape.len())));
            continue;
        }
        let row_bytes = cols * esz;
        let mut acc = StatAcc::new();
        br.seek(SeekFrom::Start(data_start + e.offs.0))
            .map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; row_bytes];
        let mut vals: Vec<f32> = vec![0.0; cols];
        for _r in 0..e.shape[0] as usize {
            br.read_exact(&mut buf)
                .map_err(|_| "short read".to_string())?;
            match e.dtype {
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
        metered += 1;
        total.merge(&acc);
        per_family
            .entry(fam)
            .or_insert_with(StatAcc::new)
            .merge(&acc);
    }
    Ok(ScanOut {
        per_family,
        total,
        skipped,
        metered_tensors: metered,
        type_counts: BTreeMap::new(),
    })
}
