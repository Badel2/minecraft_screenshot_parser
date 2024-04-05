/// Tools to detect and remove the crosshair
pub mod crosshair;
/// Tools to detect and parse a map
pub mod map;
/// Convert a map into colors from the expected map palette
pub mod map_color_correct;

pub mod deps {
    pub use image;
    pub use imageproc;
}
