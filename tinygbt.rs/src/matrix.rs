use core::ops::Index;

type Float = f64;

pub struct StridedVec<'a, A: 'a> {
    pub data: &'a [A],
    pub start: usize,
    pub stride: usize,
}

impl<'a, A: 'a> Index<usize> for StridedVec<'a, A> {
    type Output = A;
    fn index(&self, pos: usize) -> &A {
        &self.data[self.start + pos * self.stride]
    }
}

pub struct ColumnMajorMatrix {
    // Number of rows in the matrix
    pub n_rows: usize,
    // Number of columns in the matrix
    pub n_cols: usize,
    // Values used by the algorithm. Format is row first
    pub values: Vec<Float>,
}

impl ColumnMajorMatrix {
    pub fn from_columns(
        n_rows: usize,
        n_cols: usize,
        columns: &Vec<Vec<Float>>,
    ) -> ColumnMajorMatrix {
        let mut values = Vec::with_capacity(n_rows * n_cols);
        for column in columns {
            for &item in column {
                values.push(item)
            }
        }
        assert_eq!(n_rows * n_cols, values.len());
        println!("{:?}", values);
        ColumnMajorMatrix {
            n_rows,
            n_cols,
            values,
        }
    }

    pub fn column(&self, col: usize) -> &[Float] {
        let start = col * self.n_rows;
        &self.values.as_slice()[start..start + self.n_rows]
    }

    pub fn columns(&self) -> impl Iterator<Item = &[Float]> {
        self.values.chunks(self.n_rows)
    }

    pub fn row<'a>(&'a self, row: usize) -> StridedVec<Float> {
        StridedVec {
            data: &self.values,
            start: row,
            stride: self.n_rows,
        }
    }
}

impl Index<(usize, usize)> for ColumnMajorMatrix {
    type Output = Float;
    fn index(&self, (row, col): (usize, usize)) -> &Float {
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
        let matrix = ColumnMajorMatrix::from_columns(3, 2, &columns);

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
    }
}
