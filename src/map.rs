use image::DynamicImage;
use image::GenericImageView;
use image::GrayImage;
use image::Luma;
use image::Rgba;
use imageproc::rect::Rect;
use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::TryFrom;

const MAP_BORDER_IMG: &[u8] = include_bytes!("../assets/map_border_x4.png");

/// Result of detect_map function: coordinates of the map in the image, and various cropped
/// versions of the map.
pub struct DetectedMap {
    pub bounding_box: Rect,
    pub img_with_bounding_box: DynamicImage,
    pub cropped_img: DynamicImage,
    pub cropped_scaled_img: DynamicImage,
}

pub fn detect_map(original_img: &DynamicImage) -> Option<DetectedMap> {
    let template = image::load_from_memory(MAP_BORDER_IMG).unwrap();
    // Make a copy of the input image, the bounding boxes will be drawn here
    let mut img = original_img.clone();

    if let Some(scale_translate) = rectilinear_shapes_match_template(&mut img, &template) {
        let red_rectangle_template = Rect::at(140, 140).of_size(371 - 140 + 1, 371 - 140 + 1);
        let bounding_box = scale_translate.apply_rect(&red_rectangle_template);
        imageproc::drawing::draw_hollow_rect_mut(&mut img, bounding_box, Rgba([255, 0, 0, 255]));

        let cropped_img = crop_image_to_rect_with_padding(original_img, &bounding_box);
        // TODO: before scaling down, we should detect the red cross and the white player position,
        // and any other markers or corrupted pixels. A simple way to do this would be to partition
        // the cropped image into 128x128 pixels and assign a score to each pixel based on the
        // number of edges or number of different colors.

        // Scale the image to map resolution. 1 pixel in this image = 1 pixel on the map
        // This uses nearest neighbor interpolation, meaning that the output pixel is calculated
        // from one input pixel at the center of the image. It seems to be good enough, but if it
        // fails for some input try to apply a 3x3 median filter before scaling to remove noise.
        let cropped_scaled_img =
            cropped_img.resize_exact(128, 128, image::imageops::FilterType::Nearest);

        let img_with_bounding_box = img;

        Some(DetectedMap {
            bounding_box,
            cropped_img,
            cropped_scaled_img,
            img_with_bounding_box,
        })
    } else {
        None
    }
}

// TODO: this function name is a lie, it just panics on out of bounds access
fn crop_image_to_rect_with_padding(img: &DynamicImage, rect: &Rect) -> DynamicImage {
    // TODO: handle rect out of bounds by adding black pixels as padding
    img.crop_imm(
        u32::try_from(rect.left()).unwrap(),
        u32::try_from(rect.top()).unwrap(),
        rect.width(),
        rect.height(),
    )
}

// Returns the bounding box of the template in img coords. Also, draws green rectangles around the
// matches pieces of the template, and yellow rectangles around the missing pieces
pub fn rectilinear_shapes_match_template(
    img: &mut DynamicImage,
    template: &DynamicImage,
) -> Option<ScaleTranslate> {
    // TODO: currently the shapes are only extracted from the red channel
    // in other words, green and blue channels are ignored
    let template_shapes = {
        let img = template;
        let (img_r, _img_g, _img_b) = split_image_into_channels(img);

        let sobel_r = imageproc::gradients::sobel_gradients(&img_r);
        // Calculate inverse image: 255 means no border
        let inverse_sobel_thres_r = imageproc::map::map_colors(&sobel_r, |p| {
            let p = u8::try_from(p[0] / 8).unwrap();
            if p > 0 {
                Luma([0])
            } else {
                Luma([255])
            }
        });

        extract_polygons_with_no_holes(inverse_sobel_thres_r)
    };

    let img_shapes = {
        let (img_r, _img_g, _img_b) = split_image_into_channels(img);

        let sobel_r = imageproc::gradients::sobel_gradients(&img_r);
        // Calculate inverse image: 255 means no border
        let inverse_sobel_thres_r = imageproc::map::map_colors(&sobel_r, |p| {
            let p = u8::try_from(p[0] / 8).unwrap();
            if p > 0 {
                Luma([0])
            } else {
                Luma([255])
            }
        });

        extract_polygons_with_no_holes(inverse_sobel_thres_r)
    };

    // Index the img shapes by shape descriptor
    let mut shapes_map: HashMap<Vec<Direction>, Vec<(&RectPolygon, Rect)>> = HashMap::new();
    for poly in &img_shapes {
        let shape_descr = poly.shape_descriptor();
        let mut bb = poly.bounding_box();
        add_margin_to_rect(&mut bb);
        shapes_map.entry(shape_descr).or_default().push((poly, bb));
    }

    let mut best_template_to_image = None;

    // For each shape that is expected to be found, check all the shapes with this shape descriptor
    // in img_shapes
    for poly in &template_shapes {
        let shape_descr = poly.shape_descriptor();
        // Ignore rectangular shapes
        if shape_descr
            == [
                Direction::Right,
                Direction::Down,
                Direction::Left,
                Direction::Up,
            ]
        {
            continue;
        }
        let mut poly_bb = poly.bounding_box();
        add_margin_to_rect(&mut poly_bb);
        for (_cand_poly, cand_bb) in shapes_map
            .get(&shape_descr)
            .map(|x| x.as_slice())
            .unwrap_or_default()
        {
            // To check if this is a good candidate, calculate the transformation required to
            // convert template coordinates to img coordinates, and look if the other pieces are
            // where they should be
            let template_to_image = ScaleTranslate::from_rect_to_rect(&poly_bb, cand_bb);

            // Skip non-square transformations
            let scale_ratio = template_to_image.scale.0 / template_to_image.scale.1;
            if scale_ratio < 0.5 || scale_ratio > 1.5 {
                continue;
            }

            let small_bb_side = smaller_side(&cand_bb);
            let (score, found_pieces) = find_matching_pieces_using_transform(
                &shapes_map,
                &template_shapes,
                &template_to_image,
                small_bb_side,
            );
            //println!("{:?} score {}", template_to_image, score);

            // TODO: set a score limit to avoid false positives
            let min_best_score = f32::INFINITY;
            let best_score = best_template_to_image
                .as_ref()
                .map(|(score, _)| *score)
                .unwrap_or(min_best_score);
            if score < best_score {
                best_template_to_image = Some((score, (template_to_image, found_pieces)));
            }
        }
    }

    if let Some((_score, (_template_to_image, found_pieces))) = best_template_to_image {
        // Try to improve the transformation using the found_pieces
        let template_to_image = precise_scale_translate_finder(&template_shapes, &found_pieces);

        for poly in &template_shapes {
            let mut poly_bb = poly.bounding_box();
            add_margin_to_rect(&mut poly_bb);
            let piece_bb = template_to_image.apply_rect(&poly_bb);
            imageproc::drawing::draw_hollow_rect_mut(img, piece_bb, Rgba([255, 255, 0, 255]));
        }
        for found_piece in found_pieces {
            if let Some((_piece, piece_bb)) = found_piece {
                imageproc::drawing::draw_hollow_rect_mut(img, piece_bb, Rgba([0, 255, 0, 255]));
            }
        }

        return Some(template_to_image);
    }

    None
}

/// Returns the length of the smaller side of the rectangle
fn smaller_side(rect: &Rect) -> u32 {
    std::cmp::min(rect.width(), rect.height())
}

// a in [b*(1-e), b*(1+e)]
fn approx_equal<T>(a: T, b: T, e: f64) -> bool
where
    f64: TryFrom<T>,
    <f64 as TryFrom<T>>::Error: std::fmt::Display,
    T: std::fmt::Display + Copy,
{
    let a = match f64::try_from(a) {
        Ok(x) => x,
        Err(e) => panic!("Cannot convert {} to f64: {}", a, e),
    };
    let b = match f64::try_from(b) {
        Ok(x) => x,
        Err(e) => panic!("Cannot convert {} to f64: {}", b, e),
    };
    let small = b * (1.0 - e);
    let big = b * (1.0 + e);

    small <= a && a <= big
}

fn find_matching_pieces_using_transform<'a>(
    shapes_map: &HashMap<Vec<Direction>, Vec<(&'a RectPolygon, Rect)>>,
    template_shapes: &[RectPolygon],
    template_to_image: &ScaleTranslate,
    small_bb_side: u32,
) -> (f32, Vec<Option<(&'a RectPolygon, Rect)>>) {
    // TODO: this threshold should depend on scale
    let threshold = small_bb_side as f32;
    // Penalty for missing piece
    let not_found_score = 1.0;
    let mut total_score = 0.0;
    let mut found_pieces = vec![];
    for poly in template_shapes {
        found_pieces.push(None);
        let shape_descr = poly.shape_descriptor();
        let mut poly_bb = poly.bounding_box();
        add_margin_to_rect(&mut poly_bb);

        let expected_img_bb = template_to_image.apply_rect(&poly_bb);

        let mut best_piece = None;

        for (cand_poly, cand_bb) in shapes_map
            .get(&shape_descr)
            .map(|x| x.as_slice())
            .unwrap_or_default()
        {
            // The scale ratio between bounding boxes should be close to 1.0
            // Allow 90% error margin: accept scale ratios between 0.1 and 1.9
            let scale_ratio = ScaleTranslate::from_rect_to_rect(&cand_bb, &expected_img_bb);
            if !approx_equal(scale_ratio.scale.0, 1.0, 0.9) {
                continue;
            }
            if !approx_equal(scale_ratio.scale.1, 1.0, 0.9) {
                continue;
            }
            // Allow 10% error margin for the size of the smaller side of the bounding box
            if !approx_equal(smaller_side(&cand_bb), small_bb_side, 0.1) {
                continue;
            }

            // Score: distance between expected point and actual point
            let mut score = distance_between_start_of_rects(&expected_img_bb, &cand_bb);
            score += distance_between_end_of_rects(&expected_img_bb, &cand_bb);
            if best_piece.is_none() {
                best_piece = Some((score, (cand_poly, cand_bb)));
            } else {
                let best_score = best_piece.as_ref().unwrap().0;
                if score < best_score {
                    best_piece = Some((score, (cand_poly, cand_bb)));
                }
            }
        }

        if let Some((score, (cand_poly, cand_bb))) = best_piece {
            if score < threshold {
                //total_score += score;
                total_score += 0.0;
                *found_pieces.last_mut().unwrap() = Some((*cand_poly, *cand_bb));
            } else {
                total_score += not_found_score;
            }
        } else {
            total_score += not_found_score;
        }
    }

    (total_score, found_pieces)
}

/// Returns the distance between the top-left corner of two rectangles
fn distance_between_start_of_rects(a: &Rect, b: &Rect) -> f32 {
    ((a.left() as f32 - b.left() as f32).powf(2.0) + (a.top() as f32 - b.top() as f32).powf(2.0))
        .powf(0.5)
}

/// Returns the distance between the bottom-right corner of two rectangles
fn distance_between_end_of_rects(a: &Rect, b: &Rect) -> f32 {
    ((a.right() as f32 - b.right() as f32).powf(2.0)
        + (a.bottom() as f32 - b.bottom() as f32).powf(2.0))
    .powf(0.5)
}

/// Add 1 pixel of margin to the rectangle. This increases the width and height by 2 units.
fn add_margin_to_rect(rect: &mut Rect) {
    *rect = Rect::at(rect.left() - 1, rect.top() - 1).of_size(rect.width() + 2, rect.height() + 2);
    //*rect = Rect::at(rect.left() - 1, rect.top() - 1).of_size(rect.width() + 1, rect.height() + 1);
}

fn precise_scale_translate_finder(
    template_shapes: &[RectPolygon],
    found_rects: &[Option<(&RectPolygon, Rect)>],
) -> ScaleTranslate {
    assert_eq!(template_shapes.len(), found_rects.len());

    // Get 2 points from each rect, from is template and to is found points
    let mut scale_from_x: Vec<i32> = vec![];
    let mut scale_from_y: Vec<i32> = vec![];
    let mut scale_to_x: Vec<i32> = vec![];
    let mut scale_to_y: Vec<i32> = vec![];
    for i in 0..template_shapes.len() {
        if let Some((_, rect)) = found_rects[i] {
            //scale_from.push(template_shapes[i].bounding_box());
            //scale_to.push(rect);
            let mut from_bb = template_shapes[i].bounding_box();
            add_margin_to_rect(&mut from_bb);
            let points_from = rect_to_corners(&from_bb);
            let points_to = rect_to_corners(&rect);

            scale_from_x.push(points_from[0].0);
            scale_from_x.push(points_from[1].0);
            scale_from_y.push(points_from[0].1);
            scale_from_y.push(points_from[1].1);
            scale_to_x.push(points_to[0].0);
            scale_to_x.push(points_to[1].0);
            scale_to_y.push(points_to[0].1);
            scale_to_y.push(points_to[1].1);
        }
    }

    // Now find a transform that can map all the points in scale_from to scale_to
    let (scale_x, offset_x) = linear_regression(&scale_from_x, &scale_to_x);
    let (scale_y, offset_y) = linear_regression(&scale_from_y, &scale_to_y);

    let transform = ScaleTranslate {
        scale: (scale_x, scale_y),
        translate: (offset_x, offset_y),
    };

    transform
}

// TODO: the problem is that we lost the fractional part of y
// (there are no half-pixels in the screenshot)
// So we don't want to find the line with smaller error
fn linear_regression(x: &[i32], y: &[i32]) -> (f32, f32) {
    assert_eq!(x.len(), y.len());
    let mut points = HashSet::with_capacity(x.len());
    // Remove duplicate points
    for i in 0..x.len() {
        points.insert((x[i], y[i]));
    }
    let mut x = Vec::with_capacity(points.len());
    let mut y = Vec::with_capacity(points.len());
    for (px, py) in points {
        x.push(px);
        y.push(py);
    }

    let x = &x;
    let y = &y;

    // Initial guess
    let (slope_guess, intercept_guess) = simple_linear_regression(x, y);
    //let (slope, intercept) = (0.0, 0.0);
    //println!("Initial guess: {:?}", (slope_guess, intercept_guess));

    // Try to improve the guess by using gradient descent
    let num_iterations = 10_000;
    // If the learning rate is too high, it may overshoot and return (NaN, Nan)
    let learning_rate = 1e-5;
    let (slope, intercept) = gradient_descent(
        x,
        y,
        slope_guess,
        intercept_guess,
        learning_rate,
        num_iterations,
    );
    //println!("Gradient descent guess: {:?}", (slope, intercept));

    if slope.is_nan() || intercept.is_nan() {
        panic!("Gradient descent failed, try reducing the learning rate");
    }

    //let y_pred: Vec<f32> = x.iter().copied().map(|x| (x as f32) * slope + intercept).collect();
    //let y_err: Vec<i32> = y.iter().zip(y_pred.iter()).map(|(y, y_pred)| (y_pred.round() as i32 - (*y))).collect();
    //println!("Errors: {:?}", y_err);

    (slope, intercept)
}

/// Basic implementation of gradient descent with constant learning rate to try to minimize the
/// loss function J
///
/// ```ignore
/// J = ReLU((m*x + n - y)**2 - 0.25)
/// ```
///
/// This means find a linear regression but ignore errors smaller than +/- 0.5 units
fn gradient_descent(
    x: &[i32],
    y: &[i32],
    initial_m: f32,
    initial_n: f32,
    initial_learning_rate: f32,
    num_iterations: u32,
) -> (f32, f32) {
    // Rectified Linear Unit
    // Used as ReLU(x**2 - 0.25) to ignore squared errors smaller than 0.25. This expands the y
    // coordinate of each point to +/- 0.5 its observed position, and it helps the linear
    // regression so is not limited to pixel accuracy.
    // ReLU(x) == max(x, 0)
    fn relu(x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    // This function is used as the derivative of ReLU(x**2 - 0.25), because when x in range
    // [-0.5, 0.5] then x**2 - 0.25 is negative, and ReLU(x**2 - 0.25) = 0, so the derivative is
    // also 0.
    // Note that mathematically there is no derivative at x=-0.5 and x=0.5 but this implementation
    // sets it to 1 at that points.
    fn in_error_range(x: f32) -> f32 {
        if x > -0.5 && x < 0.5 {
            0.0
        } else {
            1.0
        }
    }

    //println!("Starting gradient descent with m={} n={}\nx: {:?}\ny: {:?}", initial_m, initial_n, x, y);
    assert_eq!(x.len(), y.len());
    let sample_size = x.len() as f32;
    let mut m = initial_m;
    let mut n = initial_n;
    let learning_rate = initial_learning_rate;
    for _epoch in 0..num_iterations {
        // y_pred = x * m + n
        let y_pred: Vec<f32> = x.iter().copied().map(|x| (x as f32) * m + n).collect();
        let y_err: Vec<f32> = y
            .iter()
            .zip(y_pred.iter())
            .map(|(y, y_pred)| y_pred - (*y as f32))
            .collect();
        // derivative of error with respect to n
        let dn = 1.0 / sample_size
            * (0..x.len())
                .map(|i| in_error_range(y_err[i]) * y_err[i])
                .sum::<f32>();
        // derivative of error with respect to m
        let dm = 1.0 / sample_size
            * (0..x.len())
                .map(|i| in_error_range(y_err[i]) * y_err[i] * (x[i] as f32))
                .sum::<f32>();

        let _error_value = 1.0 / (2.0 * sample_size)
            * (0..x.len())
                .map(|i| relu(y_err[i] * y_err[i] - 0.25))
                .sum::<f32>();
        //println!("Error {}: m: {} n: {}", error_value, dm, dn);

        m -= learning_rate * dm;
        n -= learning_rate * dn;
        /*
        // Ignore learning rates, just increment values by 1e-6
        // This avoids overshooting problems
        let clap_abs = |x, l: f32| {
            if x > l { l } else if x < -l { -l } else { x }
        };
        let fixed_increment_rate = 1e-6;
        let dm = clap_abs(dm, fixed_increment_rate);
        let dn = clap_abs(dn, fixed_increment_rate);
        m -= dm;
        n -= dn;
        */

        if dn == 0.0 && dm == 0.0 {
            //println!("Early return from gradient descent");
            return (m, n);
        }
    }

    (m, n)
}

fn simple_linear_regression(x: &[i32], y: &[i32]) -> (f32, f32) {
    fn mean(x: &[i32]) -> f32 {
        let mut sum = 0.0;

        for a in x {
            sum += *a as f32;
        }

        sum / x.len() as f32
    }

    let average_x = mean(x);
    let average_y = mean(y);

    let (sx2, sxy) = {
        let mut variance = 0.0;
        let mut covariance = 0.0;

        for i in 0..x.len() {
            let t = x[i] as f32 - average_x;
            let u = y[i] as f32 - average_y;
            variance += t * t;
            covariance += t * u;
        }

        (variance, covariance)
    };

    let slope = sxy / sx2;
    let intercept = average_y - slope * average_x;

    (slope, intercept)
}

fn rect_from_points<I>(points: I) -> Rect
where
    I: IntoIterator<Item = (i32, i32)>,
{
    let mut iter = points.into_iter();
    let p0 = iter.next().unwrap();
    let mut r = Rect::at(p0.0, p0.1).of_size(1, 1);

    for p in iter {
        r = expand_rect_to_point(r, p);
    }

    r
}

fn rect_to_corners(rect: &Rect) -> [(i32, i32); 2] {
    // Convert rect to 2 points, the top-left and (bottom-right + 1)
    let top_left = (rect.left(), rect.top());
    // This should be (left() + width()), or right() + 1
    let bottom_right = ((rect.right() + 1), (rect.bottom() + 1));
    //let bottom_right = ((rect.right() + 0), (rect.bottom() + 0));

    [top_left, bottom_right]
}

#[derive(Debug)]
pub struct ScaleTranslate {
    scale: (f32, f32),
    translate: (f32, f32),
}

impl ScaleTranslate {
    pub fn from_rect_to_rect(from: &Rect, to: &Rect) -> Self {
        let scale = (
            (to.width() as f32 / from.width() as f32),
            (to.height() as f32 / from.height() as f32),
        );
        let translate = (
            -from.left() as f32 * scale.0 + to.left() as f32,
            -from.top() as f32 * scale.1 + to.top() as f32,
        );

        Self { scale, translate }
    }

    pub fn apply(&self, p: (f32, f32)) -> (f32, f32) {
        (
            p.0 * self.scale.0 + self.translate.0,
            p.1 * self.scale.1 + self.translate.1,
        )
    }

    pub fn apply_rect(&self, rect: &Rect) -> Rect {
        // Convert rect to 2 points, transform these 2 points, and construct a new rect with these
        // transformed points
        let top_left = (rect.left() as f32, rect.top() as f32);
        // This should be (left() + width()), or right() + 1
        let bottom_right = ((rect.right() + 1) as f32, (rect.bottom() + 1) as f32);

        let new_top_left = self.apply(top_left);
        let new_bottom_right = self.apply(bottom_right);

        // Subtract 1 from bottom_right because it should not be inside the rect, but just outside
        let rect = rect_from_points(
            [
                (new_top_left.0.round() as i32, new_top_left.1.round() as i32),
                (
                    new_bottom_right.0.round() as i32 - 1,
                    new_bottom_right.1.round() as i32 - 1,
                ),
            ]
            .iter()
            .copied(),
        );

        rect
    }
}

// Return the pixels connected horizontally and vertically, but not the pixels connected diagonally
// This are the 4-connected neighbors
fn four_connected_to(pos: (u32, u32)) -> [(u32, u32); 4] {
    [
        Direction::Left.go_from(pos),
        Direction::Right.go_from(pos),
        Direction::Up.go_from(pos),
        Direction::Down.go_from(pos),
    ]
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum Direction {
    Left,
    Right,
    Up,
    Down,
}

impl Direction {
    // I would prefer to be able to do point.advance(direction) which is more intuitive
    fn go_from(&self, (x, y): (u32, u32)) -> (u32, u32) {
        // Use wrapping arithmetic because the coordinates are positive and small, so when doing
        // bounds checking (-1) will be seen as (u32::MAX) which will be out of bounds
        match self {
            Direction::Left => (x.wrapping_sub(1), y),
            Direction::Right => (x.wrapping_add(1), y),
            Direction::Up => (x, y.wrapping_sub(1)),
            Direction::Down => (x, y.wrapping_add(1)),
        }
    }

    fn clockwise(&self) -> Self {
        match self {
            Direction::Left => Direction::Up,
            Direction::Right => Direction::Down,
            Direction::Up => Direction::Right,
            Direction::Down => Direction::Left,
        }
    }

    fn counter_clockwise(&self) -> Self {
        // Turning counter clockwise is the same as turning clockwise 3 times
        self.clockwise().clockwise().clockwise()
    }
}

// Expand the rect so it also includes the new point
fn expand_rect_to_point(rect: Rect, point: (i32, i32)) -> Rect {
    let mut min_x = rect.left();
    let mut min_y = rect.top();
    let mut max_x = rect.right();
    let mut max_y = rect.bottom();

    let (x, y) = point;

    if x < min_x {
        min_x = x;
    }
    if x > max_x {
        max_x = x;
    }
    if y < min_y {
        min_y = y;
    }
    if y > max_y {
        max_y = y;
    }

    Rect::at(min_x, min_y).of_size((max_x - min_x) as u32 + 1, (max_y - min_y) as u32 + 1)
}

// Rectiliniar polygon represented as a starting point (any corner that can be considered top-left,
// at least locally), and a list of movements (direction and count)
// The last movement must bring the cursor back to the starting point
// If steps is empty, this is a single point
// If steps has 2 elements, this is a single line
#[derive(Debug)]
struct RectPolygon {
    start: (u32, u32),
    steps: Vec<(Direction, u32)>,
}

impl RectPolygon {
    // Return the bounding box of the RectPolygon: the smallest rectangle that includes all the
    // points from the RectPolygon
    fn bounding_box(&self) -> Rect {
        rect_from_points(self.vertices().map(|(x, y)| (x as i32, y as i32)))
    }

    fn shape_descriptor(&self) -> Vec<Direction> {
        self.steps.iter().map(|(dir, _count)| *dir).collect()
    }

    // TODO: This is wrong, it calculates the area between the center of the vertices, so a 2x2
    // polygon will have area 1 instead of the expected 4
    // https://www.mathopenref.com/coordpolygonarea2.html
    fn area(&self) -> u64 {
        let mut a: i64 = 0;
        let vertices: Vec<_> = self.vertices().collect();
        let mut j = 0;

        for i in (0..vertices.len()).rev() {
            a += (i64::from(vertices[j].0) + i64::from(vertices[i].0) + 1)
                * (i64::from(vertices[j].1) - i64::from(vertices[i].1));
            j = i;
        }

        u64::try_from(a / 2).unwrap()
    }

    fn vertices<'a>(&'a self) -> impl Iterator<Item = (u32, u32)> + 'a {
        let mut pos = self.start;

        std::iter::once(self.start).chain(self.steps.iter().cloned().filter_map(
            move |(dir, count)| {
                if count == 0 {
                    return None;
                }
                for _ in 0..count {
                    pos = dir.go_from(pos);
                }

                Some(pos)
            },
        ))
    }
}

// Input: binary image
// The input image must be pased by value because it will be overwritten during the algorithm.
// If a polygon has a hole, it is skipped and will not exist in the output.
// Uses 4-connectivity because diagonal only neighbours should be impossible using the map border
// kernels.
fn extract_polygons_with_no_holes(mut img: GrayImage) -> Vec<RectPolygon> {
    let mut rps = vec![];
    let (w, h) = img.dimensions();
    // Starting at (0, 0), go to the right and then to the next line until we hit something
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y)[0];
            if p == 0 {
                continue;
            }

            // We hit something! Because of the iteration order, this must be a top-left corner of
            // a new polygon
            //println!("Found new polygon at {:?}", (x, y));
            let rp = find_polygon_border(&img, (x, y));
            //println!("This is the polygon: {:?}\nWill delete borders now", rp);
            // Set all the pixels inside the polygon to background, to avoid detecting the same
            // polygon again. This will also check for holes, returing the number of pixels that
            // should be foreground but are not
            let num_hole_pixels = delete_polygon(&mut img, &rp);
            if num_hole_pixels > 0 {
                continue;
            }

            rps.push(rp);
        }
    }

    rps
}

// Set the color of all the pixels 4-connected to `(x, y)` to `color`.
// Returns the number of pixels that changed color.
fn bucket_tool(img: &mut GrayImage, (x, y): (u32, u32), color: u8) -> u64 {
    let (w, h) = img.dimensions();
    let mut stack = vec![(x, y)];
    let mut num_changed_pixels = 0;
    let old_color = img.get_pixel(x, y)[0];
    if color == old_color {
        return 0;
    }

    while let Some((x, y)) = stack.pop() {
        if x >= w || y >= h {
            continue;
        }

        let p = img.get_pixel(x, y)[0];
        if p != old_color {
            continue;
        }

        // Delete pixel
        img.put_pixel(x, y, Luma([color]));
        num_changed_pixels += 1;

        // Delete neighbour pixels as well
        for (nx, ny) in four_connected_to((x, y)).iter().cloned() {
            stack.push((nx, ny));
        }
    }

    num_changed_pixels
}

fn delete_connected_region(img: &mut GrayImage, (x, y): (u32, u32)) -> u64 {
    bucket_tool(img, (x, y), 0)
}

fn delete_polygon(img: &mut GrayImage, rp: &RectPolygon) -> u64 {
    let num_deleted_pixels = delete_connected_region(img, rp.start);

    // TODO: the area implementation is not correct, but it will always be smaller than the real
    // area, so if the hole size is big enough we can still detect it
    let expected_polygon_area = rp.area();
    let num_hole_pixels = expected_polygon_area.saturating_sub(num_deleted_pixels);

    num_hole_pixels
}

fn advance_steps(start: (u32, u32), dir: Direction, steps: u32) -> (u32, u32) {
    let mut pos = start;
    // The compiler can optimize this for loop into a single add
    for _ in 0..steps {
        pos = dir.go_from(pos);
    }

    pos
}

fn find_polygon_border(img: &GrayImage, start: (u32, u32)) -> RectPolygon {
    let mut steps = vec![];
    let mut pos = start;
    let mut s = (Direction::Right, 0);
    let first_step = follow_direction_while_foreground(&img, pos, s.0);
    s.1 = first_step;
    // Skip steps with length 0
    if s.1 != 0 {
        steps.push(s);
    }

    pos = advance_steps(pos, s.0, s.1);
    s = (s.0.clockwise(), 0);

    loop {
        //println!("pos {:?} dir {:?} steps: {:?}", pos, s.0, steps);
        //if steps.len() > 10 { panic!("debug"); }
        // Nothing more to the right, check down but as soon as we can, go to the right again
        let (step, new_dir) = follow_direction_until_can_turn_counter_clockwise(&img, pos, s.0);
        //println!("step {} new_dir {:?}", step, new_dir);
        s.1 = step;
        if s.1 != 0 {
            steps.push(s);
        }

        pos = advance_steps(pos, s.0, s.1);
        s = (new_dir, 0);

        // Check if we are back at the starting point
        if s.0 == Direction::Right && pos == start {
            return RectPolygon { start, steps };
        }
    }
}

// Returns number of steps and the next direction to follow. 0 means that the first pixel to the
// right of pos is a 0
fn follow_direction_until_can_turn_counter_clockwise(
    img: &GrayImage,
    mut pos: (u32, u32),
    direction: Direction,
) -> (u32, Direction) {
    let turn_direction = direction.counter_clockwise();
    let mut steps = 0;

    loop {
        if steps > 0 {
            let new_pos_2 = turn_direction.go_from(pos);
            let new_pixel =
                if new_pos_2.0 >= img.dimensions().0 || new_pos_2.1 >= img.dimensions().1 {
                    0
                } else {
                    img.get_pixel(new_pos_2.0, new_pos_2.1)[0]
                };
            if new_pixel == 255 {
                return (steps, turn_direction);
            }
        }

        let new_pos = direction.go_from(pos);
        let new_pixel = if new_pos.0 >= img.dimensions().0 || new_pos.1 >= img.dimensions().1 {
            0
        } else {
            img.get_pixel(new_pos.0, new_pos.1)[0]
        };
        if new_pixel == 0 {
            return (steps, direction.clockwise());
        }

        pos = new_pos;
        steps += 1;
    }
}

// Returns number of steps. 0 means that the first pixel to the right of pos is a 0
fn follow_direction_while_foreground(
    img: &GrayImage,
    mut pos: (u32, u32),
    direction: Direction,
) -> u32 {
    let mut steps = 0;

    loop {
        let new_pos = direction.go_from(pos);
        let new_pixel = if new_pos.0 >= img.dimensions().0 || new_pos.1 >= img.dimensions().1 {
            0
        } else {
            img.get_pixel(new_pos.0, new_pos.1)[0]
        };
        if new_pixel == 0 {
            return steps;
        }
        pos = new_pos;
        steps += 1;
    }
}

/// Split one Rgba<u8> image into 3 Luma<u8> images, one per channel: red, green and blue
fn split_image_into_channels<I>(img: &I) -> (GrayImage, GrayImage, GrayImage)
where
    I: GenericImageView<Pixel = Rgba<u8>>,
{
    let (w, h) = img.dimensions();
    let mut red_image = DynamicImage::new_luma8(w, h).to_luma8();
    let mut green_image = DynamicImage::new_luma8(w, h).to_luma8();
    let mut blue_image = DynamicImage::new_luma8(w, h).to_luma8();

    for (x, y, pixel) in img.pixels() {
        let red = Luma([pixel[0]]);
        let green = Luma([pixel[1]]);
        let blue = Luma([pixel[2]]);
        red_image.put_pixel(x, y, red);
        green_image.put_pixel(x, y, green);
        blue_image.put_pixel(x, y, blue);
    }

    (red_image, green_image, blue_image)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_translate() {
        let img_rect = Rect::at(50, 50).of_size(100, 100);
        let template_rect = Rect::at(1, 1).of_size(10, 10);
        let template_to_image = ScaleTranslate::from_rect_to_rect(&template_rect, &img_rect);
        let image_to_template = ScaleTranslate::from_rect_to_rect(&img_rect, &template_rect);

        assert_eq!(template_to_image.apply((1.0, 1.0)), (50.0, 50.0));
        assert_eq!(template_to_image.apply((11.0, 11.0)), (150.0, 150.0));
        assert_eq!(template_to_image.apply((5.0, 7.0)), (90.0, 110.0));

        assert_eq!(image_to_template.apply((50.0, 50.0)), (1.0, 1.0));
        assert_eq!(image_to_template.apply((150.0, 150.0)), (11.0, 11.0));
        assert_eq!(image_to_template.apply((90.0, 110.0)), (5.0, 7.0));

        assert_eq!(template_to_image.apply_rect(&template_rect), img_rect);
    }

    #[test]
    fn linear_regression_rounded_y() {
        let mut errors = 0;
        //for m in 0..100000 {
        //let m = 1.0 + m as f32 / 1e4;
        let m = 1.055;
        let n = 0.0;
        let x: Vec<i32> = (1..=10).collect();
        let y_exact: Vec<f32> = x.iter().copied().map(|x| m * x as f32 + n).collect();
        let y_round: Vec<i32> = y_exact.iter().map(|y| y.round() as i32).collect();
        //let y_round: Vec<i32> = y_exact.iter().map(|y| y.floor() as i32).collect();
        let (guess_m, guess_n) = linear_regression(&x, &y_round);
        let y_guess: Vec<f32> = x
            .iter()
            .copied()
            .map(|x| guess_m * x as f32 + guess_n)
            .collect();
        let y_guess_round: Vec<i32> = y_guess.iter().copied().map(|y| y.round() as i32).collect();
        let y_err: Vec<i32> = y_guess_round
            .iter()
            .copied()
            .zip(y_round.iter().copied())
            .map(|(guess, exact)| exact - guess)
            .collect();

        let good = y_err.iter().all(|y| *y == 0);

        errors += if good { 0 } else { 1 };

        assert_eq!(errors, 0);
    }

    #[test]
    fn gradient_descent_explosion() {
        // This test fails when the learning rate is set to 1e-4
        let x = vec![
            296, 220, 252, 248, 192, 380, 268, 148, 172, 140, 212, 292, 336, 372, 304, 132, 216,
            264, 356,
        ];
        let y = vec![
            638, 370, 483, 469, 271, 935, 539, 116, 201, 88, 342, 624, 779, 906, 666, 60, 356, 525,
            850,
        ];
        let (slope, intercept) = (3.526963, -405.86432);

        let learning_rate = 1e-6;
        let num_iterations = 1_000;
        let (slope, intercept) =
            gradient_descent(&x, &y, slope, intercept, learning_rate, num_iterations);

        let y_pred: Vec<f32> = x
            .iter()
            .copied()
            .map(|x| (x as f32) * slope + intercept)
            .collect();
        let y_err: Vec<i32> = y
            .iter()
            .zip(y_pred.iter())
            .map(|(y, y_pred)| (y_pred.round() as i32 - (*y)))
            .collect();

        assert_eq!(y_err, vec![0; y_err.len()]);
    }

    #[test]
    // TODO: is this case even solvable?
    // As a workaround, try to add 0.5 when converting pixel coordinates to floats: the pixel at
    // (0, 0) is actually the pixel at (0.5, 0.5) because if you scale it x2 it will be (1, 1)
    // which is the line in between pixels 0.5 and 1.5.
    #[ignore]
    fn gradient_descent_bad_case_1() {
        let x = vec![
            272, 240, 380, 372, 204, 244, 296, 180, 132, 292, 348, 140, 264, 216, 212,
        ];
        let y = vec![
            1016, 904, 1397, 1369, 777, 918, 1101, 692, 523, 1087, 1285, 551, 988, 819, 805,
        ];
        let (slope, intercept) = (3.525488, 57.55664);

        let learning_rate = 1e-6;
        let num_iterations = 1_000;
        gradient_descent(&x, &y, slope, intercept, learning_rate, num_iterations);

        let y_pred: Vec<f32> = x
            .iter()
            .copied()
            .map(|x| (x as f32) * slope + intercept)
            .collect();
        let y_err: Vec<i32> = y
            .iter()
            .zip(y_pred.iter())
            .map(|(y, y_pred)| (y_pred.round() as i32 - (*y)))
            .collect();

        assert_eq!(y_err, vec![0; y_err.len()]);
    }
}
