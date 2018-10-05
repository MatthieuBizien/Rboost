use std::sync::Arc;

type Float = f32;
type Target = f64;

pub struct Matrix {
    // Number of rows in the matrix
    pub num_rows: usize,
    // Number of columns in the matrix
    pub num_cols: usize,
    // Values used by the algorithm. Format is row first
    pub values: Arc<Vec<Float>>,
    // Labels that are looked after
    pub labels: Arc<Vec<Target>>,
    // Rows that are chosen on this split.
    pub rows: Vec<usize>,
    // Active columns
    pub cols: Arc<Vec<usize>>,
}

impl Matrix {
    pub fn new(rows: Vec<Vec<Float>>, labels: Vec<Target>) -> Matrix {
        let mut values = Vec::new();
        let num_cols = rows[0].len();
        let num_rows = rows.len();
        for row in rows.iter() {
            assert_eq!(row.len(), num_cols);
            values.extend(row.iter())
        }
        Matrix {
            num_rows,
            num_cols,
            values: Arc::new(values),
            labels: Arc::new(labels),
            rows: (0..num_rows).collect(),
            cols: Arc::new((0..num_cols).collect()),
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Float {
        let idx = row * self.num_cols + col;
        self.values[idx]
    }

    pub fn get_label(&self, row: usize) -> Target {
        return self.labels[row].clone();
    }

    pub fn filter(&self, rows: Vec<usize>) -> Matrix {
        Matrix {
            num_rows: self.num_rows,
            num_cols: self.num_cols,
            values: self.values.clone(),
            labels: self.labels.clone(),
            rows,
            cols: self.cols.clone(),
        }
    }

    /// Return all the elements and positions of the column ordered by value
    pub fn sorted_col(&self, col: usize) -> (Vec<usize>, Vec<Float>) {
        // Special vector for storing NaN
        let mut na_values = Vec::new();
        // Vector for storing regular values
        let mut values = Vec::new();

        for (n_row, &row) in self.rows.iter().enumerate() {
            let val = self.get(row, col);
            if val.is_nan() {
                na_values.push((row, val))
            } else {
                values.push((row, val))
            }
        }

        values.sort_by(|&(_, val), &(_, val2)| {
            val.partial_cmp(&val2)
                .unwrap_or(::std::cmp::Ordering::Equal)
        });
        (
            values.iter().map(|&(a, b)| a).collect(),
            values.iter().map(|&(a, b)| b).collect(),
        )
    }
}
