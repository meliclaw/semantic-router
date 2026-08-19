//! Cosine similarity and top-k — port of semantic_router/linear.py
//! Modified by Meliclaw, 2026. Original work Copyright 2024 Aurelio AI.

use ndarray::{Array1, Array2, Axis};

/// Cosine similarity between a query vector `xq` (1d) and each row of `index`.
///
/// Python: `np.dot(index, xq.T) / (norm(index, axis=1) * norm(xq))`
pub fn similarity_matrix(xq: &Array1<f32>, index: &Array2<f32>) -> Array1<f32> {
    let index_norm = index.map_axis(Axis(1), |row| row.dot(&row).sqrt());
    let xq_norm = xq.dot(xq).sqrt();
    let denom = xq_norm.max(1e-12);
    let dots = index.dot(xq);
    dots / (index_norm * denom)
}

/// Top-k scores and their original indices, returned ascending by score
/// (matches numpy argpartition + argsort in linear.py).
pub fn top_scores(sim: &Array1<f32>, top_k: usize) -> (Vec<f32>, Vec<usize>) {
    let n = sim.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let k = top_k.min(n);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        sim[a]
            .partial_cmp(&sim[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let selected = &idx[n - k..];
    let mut pairs: Vec<(f32, usize)> = selected.iter().map(|&i| (sim[i], i)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let scores = pairs.iter().map(|p| p.0).collect();
    let indices = pairs.iter().map(|p| p.1).collect();
    (scores, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn cosine_matches_python_linear_py() {
        // index = [[1,0],[0,1],[1,1]], xq = [1,0]
        // sim = [1.0, 0.0, 1/sqrt(2)]
        let index = array![[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let xq = array![1.0f32, 0.0];
        let sim = similarity_matrix(&xq, &index);
        assert_abs_diff_eq!(sim[0], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(sim[1], 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(sim[2], 1.0 / 2.0f32.sqrt(), epsilon = 1e-5);
    }

    #[test]
    fn top_scores_ascending() {
        let sim = array![0.1f32, 0.9, 0.3, 0.7];
        let (scores, idx) = top_scores(&sim, 2);
        assert_eq!(idx, vec![3, 1]);
        assert_abs_diff_eq!(scores[0], 0.7, epsilon = 1e-6);
        assert_abs_diff_eq!(scores[1], 0.9, epsilon = 1e-6);
    }
}
