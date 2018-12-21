use std::ops::Index;

/// List of list format for sparse matrix. Allows fast creation and access.
pub struct LILColumnMajorMatrix<A> {
    /// Number of rows in the matrix
    n_rows: usize,

    /// Number of columns in the matrix
    n_cols: usize,

    /// Position of the data. The first level is the columns, the second is the rows
    indices: Vec<Vec<usize>>,

    /// Values of the data. The first level is the columns, the second is the rows
    values: Vec<Vec<A>>,

    /// Null value implicitly encoded by an absence of data
    null_value: A,
}

pub struct SparseListView<'a, A: 'a> {
    /// Number of elements in the list
    len: usize,

    // Position of the data
    indices: &'a [usize],

    // Values of the data
    values: &'a [A],

    null_value: &'a A,
}

impl<A: PartialEq> LILColumnMajorMatrix<A> {
    pub fn from_columns(columns: Vec<Vec<A>>, null_value: A) -> Self {
        let (n_cols, n_rows) = (columns.len(), columns[0].len());
        let mut values = Vec::with_capacity(n_cols);
        let mut indices = Vec::with_capacity(n_cols);
        for column in columns {
            let mut col_values = Vec::new();
            let mut col_indices = Vec::new();
            for (n_row, item) in column.into_iter().enumerate() {
                if item != null_value {
                    col_indices.push(n_row);
                    col_values.push(item);
                }
            }
            values.push(col_values);
            indices.push(col_indices);
        }
        Self {
            n_rows,
            n_cols,
            values,
            indices,
            null_value,
        }
    }

    pub fn from_rows(rows: Vec<Vec<A>>, null_value: A) -> Self {
        let (n_rows, n_cols) = (rows.len(), rows[0].len());
        let mut indices: Vec<Vec<usize>> = (0..n_cols).map(|_| Vec::new()).collect();
        let mut values: Vec<Vec<A>> = (0..n_cols).map(|_| Vec::new()).collect();
        for (n_row, row) in rows.into_iter().enumerate() {
            for (n_col, elem) in row.into_iter().enumerate() {
                if elem != null_value {
                    indices[n_col].push(n_row);
                    values[n_col].push(elem);
                }
            }
        }
        Self {
            n_rows,
            n_cols,
            values,
            indices,
            null_value,
        }
    }
}

impl<A> LILColumnMajorMatrix<A> {
    pub fn column(&self, n_col: usize) -> SparseListView<A> {
        SparseListView {
            indices: &self.indices[n_col],
            values: &self.values[n_col],
            len: self.n_rows,
            null_value: &self.null_value,
        }
    }

    pub fn columns(&self) -> Vec<SparseListView<A>> {
        self.indices
            .iter()
            .zip(&self.values)
            .map(|(indices, values)| SparseListView {
                indices,
                values,
                len: self.n_rows,
                null_value: &self.null_value,
            })
            .collect()
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    pub fn null_value(&self) -> &A {
        &self.null_value
    }
}

impl<'a, A: 'a> SparseListView<'a, A> {
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &A)> {
        self.indices.iter().zip(self.values)
    }

    pub fn iter_all(&self) -> impl Iterator<Item = &A> {
        (0..self.len).map(move |idx| &self[idx])
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn null_value(&self) -> &A {
        &self.null_value
    }
}

impl<'a, A: 'a> Index<usize> for SparseListView<'a, A> {
    type Output = A;
    fn index(&self, n: usize) -> &A {
        if let Ok(idx) = self.indices.binary_search(&n) {
            &self.values[idx]
        } else {
            &self.null_value
        }
    }
}

impl<A> Index<(usize, usize)> for LILColumnMajorMatrix<A> {
    type Output = A;
    fn index(&self, (row, col): (usize, usize)) -> &A {
        if let Ok(idx) = self.indices[col].binary_search(&row) {
            &self.values[col][idx]
        } else {
            &self.null_value
        }
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
        let col0: Vec<f64> = vec![1., 2., 3.];
        let col1: Vec<f64> = vec![4., 5., 6.];
        let columns = vec![col0.clone(), col1.clone()];
        let matrix = LILColumnMajorMatrix::from_columns(columns.clone(), 1.);

        assert_eq!(matrix[(0, 0)], 1.);
        assert_eq!(matrix[(1, 0)], 2.);
        assert_eq!(matrix[(2, 0)], 3.);
        assert_eq!(matrix[(0, 1)], 4.);
        assert_eq!(matrix[(1, 1)], 5.);
        assert_eq!(matrix[(2, 1)], 6.);

        let col0_: Vec<f64> = matrix.column(0).iter_all().map(|e| *e).collect();
        let col1_: Vec<f64> = matrix.column(1).iter_all().map(|e| *e).collect();
        assert_eq!(col0, col0_);
        assert_eq!(col1, col1_);

        let tuple0: Vec<(usize, f64)> = vec![(1, 2.), (2, 3.)];
        let tuple0_: Vec<(usize, f64)> = matrix.column(0).iter().map(|(a, b)| (*a, *b)).collect();
        assert_eq!(tuple0, tuple0_);
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
        let matrix = LILColumnMajorMatrix::from_rows(rows, 1.);
        assert_eq!(matrix[(0, 0)], 1.);
        assert_eq!(matrix[(1, 0)], 2.);
        assert_eq!(matrix[(2, 0)], 3.);
        assert_eq!(matrix[(0, 1)], 4.);
        assert_eq!(matrix[(1, 1)], 5.);
        assert_eq!(matrix[(2, 1)], 6.);
    }
}
