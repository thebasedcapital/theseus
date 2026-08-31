//! stats.rs — census/conditioning statistics for theseus-scan (owner: ScanFormats).
//!
//! Mirrors inspect/src/main.rs exactly (per-family q4_block_mse mean-of-per-tensor-ratios AND
//! `_pooled`, dyn_range_log10, row_energy_imbalance, amax_over_rms, below_f16_normal,
//! frac_below_f16_normal, weights), plus a `feed_block` path for the structural quant walkers:
//! one full 32-element census block -> (amax, sum_sq, below_f16, amin_nz) folded into the same
//! accumulators as feed_row, so a structurally-scanned GGUF tensor and a dequantized one agree.
//! Streaming only: counters accumulate, nothing materializes a whole tensor.

use std::collections::BTreeMap;

pub const BLOCK: usize = 32;
pub const QBITS: f64 = 7.0; // 2^(4-1) - 1 for symmetric 4-bit
pub const F16_NORMAL_MIN: f32 = 6.103515625e-5; // 2^-14

/// Half-precision IEEE 754 binary16 -> f32 (same bit math as inspect/src/main.rs::f16_to_f32).
pub fn f16_to_f32(h: u16) -> f32 {
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

/// bfloat16 -> f32 (upper 16 bits of an f32, per inspect/src/main.rs::bf16_to_f32).
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[derive(Default, Clone)]
pub struct StatAcc {
    pub n: u64,
    pub sum_sq: f64,
    pub amax: f64,
    pub cur_block_amax_sq: f64,
    pub cur_sum_sq: f64,
    pub amin_nz: f64,
    pub below_f16: u64,
    pub block_amax_sq: f64, // sum over 32-blocks of amax^2 * BLOCK
    pub blocks: u64,
    pub row_max_e: f64,
    pub row_min_e: f64,
    pub rows: u64,
    /// per-tensor mean of the ratio: the convention the M1 predictions were registered under.
    pub tensor_ratio_sum: f64,
    pub tensor_count: u64,
}

impl StatAcc {
    pub fn new() -> StatAcc {
        StatAcc {
            amin_nz: f64::INFINITY,
            row_min_e: f64::INFINITY,
            ..Default::default()
        }
    }

    /// Dequantized path (byte-identical to inspect/src/main.rs::Acc::feed_row).
    pub fn feed_row(&mut self, row: &[f32]) {
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

    /// Structural path: one full 32-element census block, stats precomputed by a quant walker
    /// from the format's scale/q structure. `amax` is the block max|v|, `sum_sq` the block L2
    /// energy, `below` the count of 0<|v|<F16_NORMAL_MIN, `amin` the min nonzero |v|.
    #[inline]
    pub fn feed_block(&mut self, amax: f64, sum_sq: f64, below: u64, amin: f64, n: usize) {
        if n == 0 {
            return;
        }
        self.n += n as u64;
        self.sum_sq += sum_sq;
        self.cur_sum_sq += sum_sq;
        if amax > self.amax {
            self.amax = amax;
        }
        if amin > 0.0 && amin < self.amin_nz {
            self.amin_nz = amin;
        }
        self.below_f16 += below;
        // full 32-blocks are the unit the k-quants round; a short tail enters the denominator
        // but never the numerator (identical to feed_row / inspect).
        if n == BLOCK && amax > 0.0 {
            self.block_amax_sq += amax * amax * BLOCK as f64;
            self.cur_block_amax_sq += amax * amax * BLOCK as f64;
            self.blocks += 1;
        }
    }

    /// Row-boundary L2 energy (row = fastest contiguous axis), folded for row_energy_imbalance.
    #[inline]
    pub fn row_energy(&mut self, rsq: f64) {
        if rsq > 0.0 {
            self.rows += 1;
            if rsq > self.row_max_e {
                self.row_max_e = rsq;
            }
            if rsq < self.row_min_e {
                self.row_min_e = rsq;
            }
        }
    }

    pub fn close_tensor(&mut self) {
        if self.cur_sum_sq > 0.0 && self.cur_block_amax_sq > 0.0 {
            self.tensor_ratio_sum +=
                self.cur_block_amax_sq / (12.0 * QBITS * QBITS * self.cur_sum_sq);
            self.tensor_count += 1;
        }
        self.cur_block_amax_sq = 0.0;
        self.cur_sum_sq = 0.0;
    }

    pub fn merge(&mut self, o: &StatAcc) {
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

    pub fn report(&self) -> BTreeMap<&'static str, f64> {
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
        m.insert(
            "q4_block_mse_pooled",
            if self.sum_sq > 0.0 {
                self.block_amax_sq / (12.0 * QBITS * QBITS * self.sum_sq)
            } else {
                0.0
            },
        );
        m.insert(
            "q4_block_mse",
            if self.tensor_count > 0 {
                self.tensor_ratio_sum / self.tensor_count as f64
            } else {
                0.0
            },
        );
        m.insert(
            "row_energy_imbalance",
            if self.rows > 0 && self.row_min_e.is_finite() && self.row_min_e > 0.0 {
                self.row_max_e / self.row_min_e
            } else {
                0.0
            },
        );
        m.insert(
            "amax_over_rms",
            if rms > 0.0 { self.amax / rms } else { 0.0 },
        );
        m.insert("below_f16_normal", self.below_f16 as f64);
        m.insert(
            "frac_below_f16_normal",
            if self.n > 0 {
                self.below_f16 as f64 / self.n as f64
            } else {
                0.0
            },
        );
        m.insert("weights", self.n as f64);
        m
    }
}

/// Safe cosine-style membership used by tests: `a` is f64, `b` is the exposed sqrt; we use it
/// only to give the block statistic a self-check in stats-level tests.
#[cfg(test)]
pub fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}
