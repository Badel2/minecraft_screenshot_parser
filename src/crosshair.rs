use image::DynamicImage;
use image::GenericImage;
use image::GenericImageView;
use image::GrayImage;
use image::Luma;
use image::Pixel;
use image::Rgba;
use image::SubImage;
use imageproc::rect::Rect;
use std::convert::TryFrom;

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

/// Merge the input channels using median function: the output pixel will be the median value of
/// all the input pixels at the same coordinate.
fn merge_channels_pixelwise_median(channels: &[GrayImage]) -> GrayImage {
    assert_ne!(channels.len(), 0);
    // Merge the 3 channels into one
    let mut lines_red =
        DynamicImage::new_luma8(channels[0].dimensions().0, channels[0].dimensions().1).to_luma8();
    for x in 0..lines_red.dimensions().0 {
        for y in 0..lines_red.dimensions().1 {
            let mut to_be_sorted: Vec<_> = channels.iter().map(|c| c.get_pixel(x, y)[0]).collect();
            // median(p0, p1, p2) = sort([p0, p1, p2])[1]
            let sorted = {
                to_be_sorted.sort();
                to_be_sorted
            };
            let pixel = sorted[sorted.len() / 2];
            lines_red.put_pixel(x, y, Luma([pixel]));
        }
    }

    lines_red
}

/// Calculate the similarity of `next_pixel` to the inverse of `pixel`.
///
/// Returns similarity index in [0, 255] range, where 255 means exact match
/// and 0 means no match.
pub fn match_pixel_and_its_inverse(pixel: u8, next_pixel: u8) -> u8 {
    fn match_pixel_and_its_inverse_impl(pixel: u8, next_pixel: u8) -> u8 {
        let inverse = 255 - pixel;
        if pixel <= inverse {
            //let input_range = (pixel, inverse);
            let linear_start = 100;
            if pixel >= linear_start {
                // For values of pixel in range [100, 127], interpolate next_pixel from [127, inverse]
                // to [0, 255]
                let input_range = (127, inverse);
                let output_range = (0, 255);
                interpolate(next_pixel, input_range, output_range)
            } else {
                // For values of pixel in range [0, 99], interpolate next_pixel from [inverse - 28,
                // inverse] to [0, 255].
                // This means that next_pixel will be considered the inverse of pixel if it lies
                // between [inverse - 28, inverse + 28], with 100% match at inverse and 0% match at the
                // extremes.
                let input_range = (inverse - (127 - linear_start + 1), inverse);
                let output_range = (0, 255);
                interpolate(next_pixel, input_range, output_range)
            }
        } else {
            // f(a, b) is equivalent to f(255 - a, 255 - b)
            let inverse_next_pixel = 255 - next_pixel;
            match_pixel_and_its_inverse_impl(inverse, inverse_next_pixel)
        }
    }

    // Calculate both f(a, b) and f(b, a) and return the maximum
    // This is because we want the output to be equivalent, but match_pixel_and_its_inverse_impl is
    // not implemented correctly, and adding this workaround is faster than debugging
    std::cmp::max(
        match_pixel_and_its_inverse_impl(pixel, next_pixel),
        match_pixel_and_its_inverse_impl(next_pixel, pixel),
    )
}

/// Perform a kind of linear interpolation of value x from input range to output range.
///
/// `let (input_min, input_max) = input`
/// * If x is below the input range, return output_min.
/// * If x is above the input range, mirror x around input_max, so the resulting function always has
/// a maximum at x = output_max
/// * If x is inside the input range, perform linear interpolation from the input range to the
/// output range.
///
/// The ranges must be positive: input_min <= input_max
fn interpolate(x: u8, input: (u8, u8), output: (u8, u8)) -> u8 {
    assert!(input.0 <= input.1);
    assert!(output.0 <= output.1);

    if x <= input.0 {
        return output.0;
    }

    if x >= input.1 {
        if x == input.1 {
            return output.1;
        }
        // Mirror around input.1
        return interpolate(input.1 - (x - input.1), input, output);
    }

    // x from [input.0, input.1] to [0, input.range]
    let x0 = x - input.0;
    // x from [0, input.range] to [0, output.range * input.range]
    let x1 = u16::from(x0) * u16::from(output.1 - output.0);
    // Rounding division: add half the denominator to the numerator
    // x from [0, output.range * input.range] to [0, output.range * input.range / input.range]
    let x2 = (x1 + u16::from((input.1 - input.0) / 2)) / u16::from(input.1 - input.0);
    // x from [0, output.range] to [output.0, output.1]
    let x3 = x2 + u16::from(output.0);

    u8::try_from(x3).unwrap()
}

fn invert_crosshair<I>(img: &mut I, scale: u8)
where
    I: GenericImage<Pixel = Rgba<u8>>,
{
    let mut invert_pixel = |x, y| {
        let mut pixel_at_center = img.get_pixel(x, y);
        pixel_at_center.invert();
        img.put_pixel(x, y, pixel_at_center);
    };

    match scale {
        1..=6 => {
            let scale = scale.into();
            for x in 0..scale {
                for y in 0..scale * 5 {
                    invert_pixel(4 * scale + x, y);
                }
            }

            for x in 0..scale {
                for y in scale..scale * 5 {
                    invert_pixel(4 * scale + x, 4 * scale + y);
                }
            }

            for x in 0..scale * 4 {
                for y in 0..scale {
                    invert_pixel(x, scale * 4 + y);
                }
            }

            for x in scale..scale * 5 {
                for y in 0..scale {
                    invert_pixel(4 * scale + x, 4 * scale + y);
                }
            }
        }
        x => panic!("Invalid scale {}", x),
    }
}

fn find_negative_color_edges(center_red: &GrayImage) -> GrayImage {
    // Combine horizontal and vertical lines using a simple max function
    // This will ignore any diagonal edges that we are not interested in
    let mut lines_red =
        DynamicImage::new_luma8(center_red.dimensions().0 - 1, center_red.dimensions().1 - 1)
            .to_luma8();
    for x in 0..lines_red.dimensions().0 {
        for y in 0..lines_red.dimensions().1 {
            let p0 = center_red.get_pixel(x, y)[0];
            let p1 = center_red.get_pixel(x, y + 1)[0];
            let p2 = center_red.get_pixel(x + 1, y)[0];
            let p3 = center_red.get_pixel(x + 1, y + 1)[0];

            let p01 = match_pixel_and_its_inverse(p0, p1);
            let p21 = match_pixel_and_its_inverse(p2, p1);
            let p03 = match_pixel_and_its_inverse(p0, p3);
            let p23 = match_pixel_and_its_inverse(p2, p3);
            let ph = u8::try_from(
                (u16::from(p01) + u16::from(p21) + u16::from(p03) + u16::from(p23)) / 4,
            )
            .unwrap();

            let p02 = match_pixel_and_its_inverse(p0, p2);
            let p12 = p21;
            let p13 = match_pixel_and_its_inverse(p1, p3);
            let pv = u8::try_from(
                (u16::from(p02) + u16::from(p12) + u16::from(p03) + u16::from(p13)) / 4,
            )
            .unwrap();

            // ph is the horizontal edge, pv is the vertical edge
            // max(ph, pv)
            let pixel = if ph > pv { ph } else { pv };
            lines_red.put_pixel(x, y, Luma([pixel]));
        }
    }

    lines_red
}

fn crop_image_at_center(
    img: &mut DynamicImage,
    center_radius: u32,
) -> (SubImage<&mut DynamicImage>, (u32, u32, u32, u32)) {
    let (w, h) = img.dimensions();
    let (center_x, center_y) = ((w - 1) / 2, (h - 1) / 2);
    let center_offset_x = center_x.saturating_sub(center_radius);
    let center_offset_y = center_y.saturating_sub(center_radius);
    let center_w = center_radius * 2 + 1;
    let max_center_w = w - center_offset_x;
    let center_w = std::cmp::min(center_w, max_center_w);
    let center_h = center_radius * 2 + 1;
    let max_center_h = h - center_offset_y;
    let center_h = std::cmp::min(center_h, max_center_h);

    // Warning: sub_image does not perform bounds checking!
    // This means that if center_img has dimensions (10, 10) and you do
    // center_img.get_pixel(100, 100), it will get the pixels from the original image!
    let center_img = img.sub_image(center_offset_x, center_offset_y, center_w, center_h);

    (
        center_img,
        (center_offset_x, center_offset_y, center_w, center_h),
    )
}

lazy_static::lazy_static! {
    static ref CROSSHAIR_KERNELS: [GrayImage; 6] = generate_crosshair_kernels_6();
}

fn get_crosshair_kernels() -> &'static [GrayImage; 6] {
    &CROSSHAIR_KERNELS
}

fn generate_crosshair_kernel(scale: u8) -> GrayImage {
    // 10, 19, 28, 37
    let size = (1 + scale * 9) as u32;
    // 0, 1, 2, 3
    let top = (scale - 1) as u32;
    // 5, 9, 13, 17
    let arm = (1 + scale * 4) as u32;

    let white = Luma([255]);
    let mut gray = GrayImage::from_raw(size, size, vec![0; size as usize * size as usize]).unwrap();
    {
        let gray = &mut gray;

        let draw_hline = |gray: &mut GrayImage, (x0, x1), y| {
            for x in x0..x1 {
                gray.put_pixel(x, y, white);
            }
        };
        let draw_vline = |gray: &mut GrayImage, x, (y0, y1)| {
            for y in y0..y1 {
                gray.put_pixel(x, y, white);
            }
        };

        // Draw top and bottom
        draw_hline(gray, (arm, arm + top), 0);
        draw_hline(gray, (arm, arm + top), size - 1);

        // Draw left and right
        draw_vline(gray, 0, (arm, arm + top));
        draw_vline(gray, size - 1, (arm, arm + top));

        // Draw top and bottom arm, left side
        draw_vline(gray, arm - 1, (0, arm));
        draw_vline(gray, arm - 1, (arm + top, size));

        // Draw top and bottom arm, right side
        draw_vline(gray, arm + top, (0, arm));
        draw_vline(gray, arm + top, (arm + top, size));

        // Draw left and right arm, top side
        draw_hline(gray, (0, arm), arm - 1);
        draw_hline(gray, (arm + top, size), arm - 1);

        // Draw left and right arm, bottom side
        draw_hline(gray, (0, arm), arm + top);
        draw_hline(gray, (arm + top, size), arm + top);
    }

    gray
}

fn generate_crosshair_kernels_6() -> [GrayImage; 6] {
    [
        generate_crosshair_kernel(1),
        generate_crosshair_kernel(2),
        generate_crosshair_kernel(3),
        generate_crosshair_kernel(4),
        generate_crosshair_kernel(5),
        generate_crosshair_kernel(6),
    ]
}

#[derive(Debug)]
struct ScoreInfo {
    score: u64,
    second_score: u64,
    worst_score: u64,
    num_edges_before: u32,
    num_edges_after: u32,
}

#[derive(Debug)]
pub struct Crosshair {
    scale: u8,
    top_left: (u32, u32),
    score_info: ScoreInfo,
}

impl Crosshair {
    pub fn scale(&self) -> u8 {
        self.scale
    }

    pub fn top_left(&self) -> (u32, u32) {
        self.top_left
    }
}

// Try to detect the crosshair.
// The crosshair inverts the color of the pixels where it is drawn, and it is always drawn at
// the center of the image.
// TODO: the crosshair can be changed using texture packs
pub fn remove_crosshair(img: &mut DynamicImage) -> Option<Crosshair> {
    // Magic constants:
    // Draw bounding box
    //let center_radius = 6;
    let center_radius = 60;

    let (w, h) = img.dimensions();
    if w <= 2 || h <= 2 {
        // Error: input image too small
        return None;
    }

    let (mut center_img, (center_offset_x, center_offset_y, _center_w, _center_h)) =
        crop_image_at_center(img, center_radius);

    let (center_red, center_green, center_blue) = split_image_into_channels(&*center_img);

    let mut channels = Vec::with_capacity(3);
    for center_red in [center_red, center_green, center_blue] {
        let lines_red = find_negative_color_edges(&center_red);
        channels.push(lines_red);
    }

    // Merge the 3 channels into one
    let lines_red = merge_channels_pixelwise_median(&channels);

    let crosshair_kernels = get_crosshair_kernels();
    // Correlate crosshair kernel with inverse_score matrix
    let percent_good = [87, 85, 83, 81, 79, 77];

    for i in (0..crosshair_kernels.len()).rev() {
        let scale = i + 1;
        // TODO: make sure this never panics
        let crosshair_score = imageproc::template_matching::match_template(
            &lines_red,
            &crosshair_kernels[i],
            imageproc::template_matching::MatchTemplateMethod::CrossCorrelation,
        );

        let mut score_map: Vec<(u64, (usize, usize))> = crosshair_score
            .enumerate_pixels()
            .map(|(x, y, pixel_luma)| (pixel_luma[0] as u64, (x as usize, y as usize)))
            .collect();
        // TODO: it is possible to get the elements (0, 1, last) without sorting
        score_map.sort_by_key(|k| {
            // if MatchTemplateMethod == CrossCorrelation, high score is better
            std::cmp::Reverse(*k)
        });

        let (max_pixel_value, _) = score_map.last().unwrap();
        let (min_pixel_value, (min_x, min_y)) = score_map[0];
        let (second_min_pixel_value, _) = score_map[1];

        // Found crosshair!
        //if min_pixel_value >= crosshair_scores[i] {
        if min_pixel_value > 0 && second_min_pixel_value * 100 / min_pixel_value < percent_good[i] {
            let crosshair_size = match scale {
                1..=6 => (scale * 9 + 2) as u32,
                /*
                1 => 11,
                2 => 20,
                3 => 29,
                4 => 38,
                */
                x => panic!("Invalid crosshair scale: {}", x),
            };
            let rect = Rect::at(min_x as i32, min_y as i32).of_size(crosshair_size, crosshair_size);

            // Check if inverting the crosshair will decrease the number of edges in the image
            let num_edges_before = lines_red.pixels().fold(0u32, |acc, x| acc + x[0] as u32);

            // Invert crosshair
            invert_crosshair(
                &mut *center_img.sub_image(
                    rect.left() as u32 + 1,
                    rect.top() as u32 + 1,
                    rect.width() - 2,
                    rect.height() - 2,
                ),
                scale as u8,
            );

            let (center_red, center_green, center_blue) = split_image_into_channels(&*center_img);

            let mut channels = vec![];
            for center_red in [center_red, center_green, center_blue] {
                let lines_red = find_negative_color_edges(&center_red);
                channels.push(lines_red);
            }

            // Merge the 3 channels into one
            let lines_red = merge_channels_pixelwise_median(&channels);

            let num_edges_after = lines_red.pixels().fold(0u32, |acc, x| acc + x[0] as u32);

            if num_edges_after < num_edges_before {
                // Found!

                // Stop looking for smaller crosshairs because they will probably also match the
                // template
                return Some(Crosshair {
                    scale: scale as u8,
                    top_left: (
                        rect.left() as u32 + 1 + center_offset_x,
                        rect.top() as u32 + 1 + center_offset_y,
                    ),
                    score_info: ScoreInfo {
                        score: min_pixel_value,
                        second_score: second_min_pixel_value,
                        worst_score: *max_pixel_value,
                        num_edges_before,
                        num_edges_after,
                    },
                });
            } else {
                // Invert crosshair back and keep looking for smaller crosshairs
                invert_crosshair(
                    &mut *center_img.sub_image(
                        rect.left() as u32 + 1,
                        rect.top() as u32 + 1,
                        rect.width() - 2,
                        rect.height() - 2,
                    ),
                    scale as u8,
                );
                continue;
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_crosshair_kernels() {
        fn generate_crosshair_kernels() -> [GrayImage; 4] {
            [
                generate_crosshair_kernel(1),
                generate_crosshair_kernel(2),
                generate_crosshair_kernel(3),
                generate_crosshair_kernel(4),
            ]
        }

        let crosshair_kernel1 = GrayImage::from_raw(
            10,
            10,
            vec![
                0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0,
                255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0,
                0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0,
                0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0,
            ],
        )
        .unwrap();
        let crosshair_kernel2 = GrayImage::from_raw(
            19,
            19,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0,
                0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255,
                0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0,
                255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        )
        .unwrap();
        let crosshair_kernel3 = GrayImage::from_raw(
            28,
            28,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0,
                255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255,
                0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0,
                255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255,
                0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        )
        .unwrap();
        let crosshair_kernel4 = GrayImage::from_raw(
            37,
            37,
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0,
                255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0,
                255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0,
                255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255,
                255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255,
                0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0,
                0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0,
                0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        )
        .unwrap();
        let cr = [
            crosshair_kernel1,
            crosshair_kernel2,
            crosshair_kernel3,
            crosshair_kernel4,
        ];
        let cs = generate_crosshair_kernels();

        imageproc::assert_pixels_eq!(cs[0], cr[0]);
        imageproc::assert_pixels_eq!(cs[1], cr[1]);
        imageproc::assert_pixels_eq!(cs[2], cr[2]);
        imageproc::assert_pixels_eq!(cs[3], cr[3]);
        assert_eq!(cr, cs);
    }
}
