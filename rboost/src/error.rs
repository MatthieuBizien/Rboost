use std::{error, fmt};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Generic error type on fitting error.
///
/// We have to define a specific type because the generic dyn Error is not Sync, so it can't be used
/// with Rayon.
pub struct FitError {
    msg: String,
}

impl FitError {
    fn new(msg: String) -> FitError {
        FitError { msg }
    }
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("invalid first item to double")
    }
}

// This is important for other errors to wrap this one.
impl error::Error for FitError {
    fn description(&self) -> &str {
        &self.msg
    }

    fn cause(&self) -> Option<&error::Error> {
        // Generic error, underlying cause isn't tracked.
        None
    }
}

impl std::convert::From<&str> for FitError {
    fn from(msg: &str) -> Self {
        FitError::new(msg.to_string())
    }
}

impl std::convert::From<String> for FitError {
    fn from(msg: String) -> Self {
        FitError::new(msg)
    }
}

pub type FitResult<T> = Result<T, FitError>;

pub(crate) static SHOULD_NOT_HAPPEN: &str =
    "There is an unexpected error in Rboost. Please raise a bug.";
