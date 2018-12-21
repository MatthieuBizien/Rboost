use core::ops::Index;

/// Slice of data with a stride.
pub struct StridedVecView<'a, A: 'a> {
    pub data: &'a [A],
    pub start: usize,
    pub stride: usize,
}

impl<'a, A: 'a> StridedVecView<'a, A> {
    pub fn new(data: &'a [A], start: usize, stride: usize) -> Self {
        Self {
            data,
            start,
            stride,
        }
    }

    pub fn from_slice(data: &'a [A]) -> Self {
        Self {
            data,
            start: 0,
            stride: 1,
        }
    }
}

impl<'a, A: 'a> Index<usize> for StridedVecView<'a, A> {
    type Output = A;
    fn index(&self, pos: usize) -> &A {
        &self.data[self.start + pos * self.stride]
    }
}

impl<'a, A: 'a> StridedVecView<'a, A> {
    pub fn iter(&'a self) -> impl Iterator<Item = &A> {
        let n_iter = self.data.len() / self.stride;
        //(0..n_iter).map(move |pos| &self[pos])
        (0..n_iter).map(move |pos| &self[pos])
    }
}

/// Store a dense matrix in a column-major way.
pub struct ColumnMajorMatrix<A> {
    /// Number of rows in the matrix
    n_rows: usize,
    /// Number of columns in the matrix
    n_cols: usize,
    /// Values used by the algorithm. Format is row first
    values: Vec<A>,
}

impl<A> ColumnMajorMatrix<A> {
    pub fn from_columns(columns: Vec<Vec<A>>) -> Self {
        let (n_cols, n_rows) = (columns.len(), columns[0].len());
        let mut values = Vec::with_capacity(n_rows * n_cols);
        for column in columns {
            for item in column {
                values.push(item)
            }
        }
        assert_eq!(n_rows * n_cols, values.len());
        Self {
            n_rows,
            n_cols,
            values,
        }
    }

    pub fn from_rows(rows: Vec<Vec<A>>) -> Self {
        let (n_rows, n_cols) = (rows.len(), rows[0].len());
        let mut values: Vec<A> = Vec::with_capacity(n_rows * n_cols);
        let mut rows: Vec<_> = rows.into_iter().map(|c| c.into_iter()).collect();
        loop {
            let mut n_ko = 0;
            for row in &mut rows {
                if let Some(item) = row.next() {
                    values.push(item)
                } else {
                    n_ko += 1;
                }
            }
            if n_ko > 0 {
                assert_eq!(n_ko, n_rows);
                break;
            }
        }
        assert_eq!(n_rows * n_cols, values.len());
        Self {
            n_rows,
            n_cols,
            values,
        }
    }

    pub fn from_function(n_rows: usize, n_cols: usize, f: impl Fn(usize, usize) -> A) -> Self {
        let mut values = Vec::new();
        for col in 0..n_cols {
            for row in 0..n_rows {
                values.push(f(row, col));
            }
        }
        Self {
            n_rows,
            n_cols,
            values,
        }
    }

    pub fn column(&self, col: usize) -> &[A] {
        let start = col * self.n_rows;
        &self.values.as_slice()[start..start + self.n_rows]
    }

    pub fn column_mut(&mut self, col: usize) -> &mut [A] {
        let start = col * self.n_rows;
        &mut self.values.as_mut_slice()[start..start + self.n_rows]
    }

    pub fn columns(&self) -> impl Iterator<Item = &[A]> {
        self.values.chunks(self.n_rows)
    }

    pub fn row(&self, row: usize) -> StridedVecView<A> {
        StridedVecView::new(&self.values, row, self.n_rows)
    }

    pub fn flat(&self) -> &Vec<A> {
        &self.values
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }
}

impl<A> Index<(usize, usize)> for ColumnMajorMatrix<A> {
    type Output = A;
    fn index(&self, (row, col): (usize, usize)) -> &A {
        // No need to check for col because it fill be out of the buffer
        assert!(row < self.n_rows);
        &self.values[row + col * self.n_rows]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_column_major() {
        // 1 4
        // 2 5
        // 3 6
        let col0 = vec![1., 2., 3.];
        let col1 = vec![4., 5., 6.];
        let columns = vec![col0.clone(), col1.clone()];
        let matrix = ColumnMajorMatrix::from_columns(columns.clone());

        assert_eq!(matrix[(0, 0)], 1.);
        assert_eq!(matrix[(1, 0)], 2.);
        assert_eq!(matrix[(2, 0)], 3.);
        assert_eq!(matrix[(0, 1)], 4.);
        assert_eq!(matrix[(1, 1)], 5.);
        assert_eq!(matrix[(2, 1)], 6.);

        assert_eq!(col0, matrix.column(0));
        assert_eq!(col1, matrix.column(1));

        assert_eq!(columns, matrix.columns().collect::<Vec<_>>());

        assert_eq!(matrix.row(0)[0], 1.);
        assert_eq!(matrix.row(1)[0], 2.);
        assert_eq!(matrix.row(2)[0], 3.);
        assert_eq!(matrix.row(0)[1], 4.);
        assert_eq!(matrix.row(1)[1], 5.);
        assert_eq!(matrix.row(2)[1], 6.);

        let row0: Vec<f64> = matrix.row(0).iter().map(|&e| e.clone()).collect();
        let row1: Vec<f64> = matrix.row(1).iter().map(|&e| e.clone()).collect();
        let row2: Vec<f64> = matrix.row(2).iter().map(|&e| e.clone()).collect();
        assert_eq!(row0, vec![1., 4.]);
        assert_eq!(row1, vec![2., 5.]);
        assert_eq!(row2, vec![3., 6.]);
    }

    #[test]
    fn test_column_major_from_rows() {
        // 1 4
        // 2 5
        // 3 6
        let row0 = vec![1., 4.];
        let row1 = vec![2., 5.];
        let row2 = vec![3., 6.];
        let rows = vec![row0.clone(), row1.clone(), row2.clone()];
        let matrix = ColumnMajorMatrix::from_rows(rows);
        assert_eq!(matrix[(0, 0)], 1.);
        assert_eq!(matrix[(1, 0)], 2.);
        assert_eq!(matrix[(2, 0)], 3.);
        assert_eq!(matrix[(0, 1)], 4.);
        assert_eq!(matrix[(1, 1)], 5.);
        assert_eq!(matrix[(2, 1)], 6.);
    }
}
