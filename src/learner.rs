use data::Matrix;

type Float = f32;
type Target = f64;

struct SplitNode {
    // Which feature we split on
    col: usize,
    // The maximum (included) value for the left branch
    limit: Float,
    left_child: Node,
    right_child: Node,
}

enum Node {
    Split(Box<SplitNode>),
    FinalNode(Target),
}

struct Split {
    left_child: Matrix,
    right_child: Matrix,
}

impl SplitNode {
    fn split(&self, matrix: Matrix) -> Split {
        let mut left_rows = Vec::new();
        let mut right_rows = Vec::new();
        for &row in &matrix.rows {
            let val = matrix.get(row, self.col);
            if val.is_nan() {
                unimplemented!();
            } else if val <= self.limit {
                left_rows.push(row)
            } else {
                right_rows.push(row)
            }
        }
        Split {
            left_child: matrix.filter(left_rows),
            right_child: matrix.filter(right_rows),
        }
    }
}
