use image::DynamicImage;
use image::GenericImage;
use image::GenericImageView;
use image::GrayImage;
use image::Luma;
use image::Rgb;
use image::Rgba;
use std::collections::HashMap;
use std::collections::HashSet;

pub fn extract_unexplored_treasure_map(img: &DynamicImage) -> GrayImage {
    // The input image must be 128x128 in size
    assert_eq!(img.dimensions(), (128, 128));

    // Split the image into odd and even rows
    // TODO: this could be used to detect water: even_img has water lines but odd_img uses land
    // color for water
    let (_even_img, odd_img) = split_img_into_rows_mod_2(img);
    // The two most common colors of odd_img will be the background color (color_land)
    let hist_odd = RgbHistogram::new(img);
    // TODO: we could improve the confidence of this check by checking that count1 is approx equal
    // to count2, and also that count3 is around count1 / 2
    let [(color_background1, _count1), (color_background2, _count2)] =
        hist_odd.top2_colors().unwrap();

    // Most of the pixels should be background, so we can use that information to calculate a light
    // mask. This is needed because the colors of the map are not constant.
    let light_mask1 = img_light_mask(&odd_img, color_background1);
    let light_mask2 = img_light_mask(&odd_img, color_background2);

    // Any color darker than (thres / 255) * background is assumed to not be background, so the
    // light mask should have no value at that point.
    let thres = 200;
    let light_mask12 = combine_light_masks(&light_mask1, &light_mask2, thres);

    // Fill the holes caused by non-background pixels: apply a median filter to the rgb image until
    // there are no more black pixels in the light mask, or until the limit is reached
    let dilation_loop_limit = 100;
    let dilated = repeated_rgb_dilation(&light_mask12, dilation_loop_limit);

    // The dilated light mask was created by using the odd rows of the image. We want to duplicate
    // the rows so that the dilated_full_width light mask has the same dimensions as the input img
    let dilated_full_width = duplicate_rows(&dilated);

    // Apply light mask to original image. This should result in an image where all the colors that
    // are similar to the human eye have similar rgb values.
    // To see an example of why is this needed, take a look at the checker shadow illusion:
    // https://en.wikipedia.org/wiki/Checker_shadow_illusion
    // Given that image we would want to segment it according to the 3 colors: green, light gray,
    // dark gray, and background. But in reality the 3 colors cannot be isolated, because some
    // color values can be either light gray or dark gray depending on where they are.
    // If you know some algorithm that solves this problem in a more reliable way, feel free to
    // open an issue or send a pull request.
    let norm_img = apply_light_mask(img, &dilated_full_width);

    // Given the light-normalized image, we can simply use the histogram to extract the different
    // colors.
    let hist_rgb = RgbHistogram::new(&norm_img);

    let background_similarity = 2;
    let similarity = 3;
    let mut hist_rgb = hist_rgb;

    // Return combined color and variant as one u8
    let cv = |color: u8, variant: u8| color * 4 + variant;
    // Expected color order when sorted by green channel:
    // shore3, shore1, water0, water1, water2, background
    // Note that water3 is very rare so it is not handled
    let palette = [
        cv(26, 3),
        cv(26, 1),
        cv(15, 0),
        cv(15, 1),
        cv(15, 2),
        cv(0, 0),
    ];

    // Extract the 6 most common colors
    // The 2 background colors are combined into 1, assuming that the 2 most common colors are
    // background
    let mut mcc: Vec<_> = std::iter::once(
        hist_rgb
            .remove_most_common_color_and_similar2(background_similarity)
            .map(|(removed_similar_colors, _top_color2)| removed_similar_colors)
            .unwrap(),
    )
    .chain((0..5).map(|_| {
        hist_rgb
            .remove_most_common_color_and_similar(similarity)
            .unwrap()
    }))
    .collect();

    if mcc.len() < palette.len() {
        // Found fewer colors than expected. This is probably not a valid treasure map.
        panic!("Found fewer unique colors than expected from an unexplored treasure map. Found {} colors, expected at least {}", mcc.len(), palette.len());
    }

    // Sort the removed colors according to green channel
    mcc.sort_unstable_by_key(
        |RemovedSimilarColors {
             top_color: rgb,
             removed_colors: _,
             total_count: _,
         }| {
            let Rgb([_r, g, _b]) = rgb;

            *g
        },
    );

    // Use colors to paint image using a grayscale palette
    let color_undefined = Luma([255u8]);
    let mut out_img = imageproc::map::map_colors(&norm_img, |_| color_undefined);
    for (
        RemovedSimilarColors {
            top_color: _,
            removed_colors: colors,
            total_count: _,
        },
        out_color,
    ) in mcc.iter().zip(palette.iter())
    {
        let colors_hs: HashSet<Rgb<u8>> = colors.iter().cloned().collect();
        extract_colors_from_image(&norm_img, &mut out_img, &colors_hs, Luma([*out_color]));
    }

    out_img
}

fn split_img_into_rows_mod_2(img: &DynamicImage) -> (DynamicImage, DynamicImage) {
    let (w, h) = img.dimensions();

    // TODO: this is supposed to create two images with the same format as img but half the height
    // But it should not use clones
    let mut even_img = img.clone();
    even_img = even_img.crop_imm(0, 0, w, h / 2);
    let mut odd_img = img.clone();
    odd_img = odd_img.crop_imm(0, 0, w, h - h / 2);

    for x in 0..w {
        for y in 0..h {
            let p = img.get_pixel(x, y);

            if y % 2 == 0 {
                &mut even_img
            } else {
                &mut odd_img
            }
            .put_pixel(x, y / 2, p);
        }
    }

    (even_img, odd_img)
}

// TODO: this function assumes that the input image height is a multiple of 2
// In some cases it may be needed to do `h * 2 - 1`
fn duplicate_rows(even_img: &DynamicImage) -> DynamicImage {
    let (w, h) = even_img.dimensions();
    let img = even_img.resize_exact(w, h * 2, image::imageops::FilterType::Nearest);

    img
}
/*
fn duplicate_rows(even_img: &DynamicImage, out: &mut DynamicImage) {
    let (w1, h1) = even_img.dimensions();
    let (w2, h2) = out.dimensions();
    assert_eq!(w1, w2);
    assert_eq!(h1 * 2, h2);

    for x in 0..w1 {
        for y in 0..h1 {
            let p = even_img.get_pixel(x, y);
            out.put_pixel(x, y * 2, p);
            out.put_pixel(x, y * 2 + 1, p);
        }
    }
}
*/

struct RemovedSimilarColors {
    pub top_color: Rgb<u8>,
    pub removed_colors: Vec<Rgb<u8>>,
    pub total_count: u32,
}

struct RgbHistogram {
    color_count: HashMap<Rgb<u8>, u32>,
}

impl RgbHistogram {
    // Count the ocurrences of each color in an image
    fn new(img: &DynamicImage) -> Self {
        let (w, h) = img.dimensions();
        let mut hg = HashMap::new();

        for x in 0..w {
            for y in 0..h {
                let Rgba([r, g, b, _a]) = img.get_pixel(x, y);
                let p = Rgb([r, g, b]);
                *hg.entry(p).or_default() += 1;
            }
        }

        Self::from_color_count(hg)
    }

    fn from_color_count(color_count: HashMap<Rgb<u8>, u32>) -> Self {
        Self { color_count }
    }

    fn iter_colors_similar_to(
        &self,
        center: Rgb<u8>,
        similarity: u8,
    ) -> impl Iterator<Item = (Rgb<u8>, u32)> + '_ {
        colors_similar_to(center, similarity)
            .into_iter()
            .filter_map(move |color| {
                self.color_count
                    .get_key_value(&color)
                    .map(|(color, count)| (*color, *count))
            })
    }

    fn color_with_max_count(&self) -> Option<(Rgb<u8>, u32)> {
        self.color_count
            .iter()
            .max_by_key(|(_color, count)| *count)
            .map(|(color, count)| (*color, *count))
    }

    // Return the 2 most common colors
    fn top2_colors(&self) -> Option<[(Rgb<u8>, u32); 2]> {
        // TODO: this iterates over self.color_count 2 times when 1 time should be enough
        // Is there a max2_by_key?
        let top1 = self
            .color_count
            .iter()
            .max_by_key(|(_color, count)| *count)
            .map(|(color, count)| (*color, *count))?;
        let top2 = self
            .color_count
            .iter()
            .filter(|x| *x != (&top1.0, &top1.1))
            .max_by_key(|(_color, count)| *count)
            .map(|(color, count)| (*color, *count))?;

        Some([top1, top2])
    }

    // Find the most frequent color, any colors similar to it, and any colors similar to that
    // colors until there are no similar colors left.
    fn remove_most_common_color_and_similar(
        &mut self,
        similarity: u8,
    ) -> Option<RemovedSimilarColors> {
        let (top_color, center_count) = self.color_with_max_count()?;
        self.color_count.remove(&top_color);
        let mut removed_colors = vec![top_color];
        let mut total_count = center_count;

        let mut queue = vec![top_color];

        while let Some(top_color) = queue.pop() {
            let sim: Vec<_> = self.iter_colors_similar_to(top_color, similarity).collect();
            for (color, count) in sim {
                self.color_count.remove(&color);
                removed_colors.push(color);
                total_count += count;
                queue.push(color);
            }
        }

        Some(RemovedSimilarColors {
            top_color,
            removed_colors,
            total_count,
        })
    }

    // Find the 2 most frequent colors, any colors similar to them, and any colors similar to that
    // colors until there are no similar colors left.
    // Returns None if the image has less than 2 unique colors.
    fn remove_most_common_color_and_similar2(
        &mut self,
        similarity: u8,
    ) -> Option<(RemovedSimilarColors, Rgb<u8>)> {
        let (top_color, center_count) = self.color_with_max_count()?;
        self.color_count.remove(&top_color);
        let mut removed_colors = vec![top_color];
        let mut total_count = center_count;

        let (top2_color, center2_count) = self.color_with_max_count()?;
        self.color_count.remove(&top2_color);
        removed_colors.push(top2_color);
        total_count += center2_count;

        let mut queue = vec![top_color, top2_color];

        while let Some(top_color) = queue.pop() {
            let sim: Vec<_> = self.iter_colors_similar_to(top_color, similarity).collect();
            for (color, count) in sim {
                self.color_count.remove(&color);
                removed_colors.push(color);
                total_count += count;
                queue.push(color);
            }
        }

        Some((
            RemovedSimilarColors {
                top_color,
                removed_colors,
                total_count,
            },
            top2_color,
        ))
    }
}

fn colors_similar_to(Rgb([cr, cg, cb]): Rgb<u8>, similarity: u8) -> Vec<Rgb<u8>> {
    let mut colors = vec![];

    for r in cr.saturating_sub(similarity)..=cr.saturating_add(similarity) {
        for g in cg.saturating_sub(similarity)..=cg.saturating_add(similarity) {
            for b in cb.saturating_sub(similarity)..=cb.saturating_add(similarity) {
                if colors_similar(Rgb([cr, cg, cb]), Rgb([r, g, b]), similarity) {
                    colors.push(Rgb([r, g, b]));
                }
            }
        }
    }

    colors
}

// Return the absolute difference between two values
fn abs_diff(a: u8, b: u8) -> u8 {
    std::cmp::max(a, b) - std::cmp::min(a, b)
}

// Returns true if the two input colors are "close enough" in RGB space, using Manhattan distance
fn colors_similar(Rgb([r1, g1, b1]): Rgb<u8>, Rgb([r2, g2, b2]): Rgb<u8>, similarity: u8) -> bool {
    (abs_diff(r1, r2) + abs_diff(g1, g2) + abs_diff(b1, b2)) <= similarity
}

// For all the pixels in img with value in in_colors, write out_color to out_img
fn extract_colors_from_image(
    img: &DynamicImage,
    out_img: &mut GrayImage,
    in_colors: &HashSet<Rgb<u8>>,
    out_color: Luma<u8>,
) {
    let (w, h) = img.dimensions();

    for x in 0..w {
        for y in 0..h {
            let Rgba([r, g, b, _a]) = img.get_pixel(x, y);
            let p_rgb = Rgb([r, g, b]);
            if in_colors.contains(&p_rgb) {
                out_img.put_pixel(x, y, out_color);
            }
        }
    }
}

// Divide img pixels by mask pixels and multiply by 255
fn apply_light_mask(img: &DynamicImage, mask: &DynamicImage) -> DynamicImage {
    let (w, h) = img.dimensions();
    let mut out = img.clone();

    for x in 0..w {
        for y in 0..h {
            let apply = |c0, c1| {
                let f = (c0 as f32) / (c1 as f32) * 255.0;
                if f > 255.0 {
                    255
                } else {
                    f.round() as u8
                }
            };

            let Rgba([r0, g0, b0, _a0]) = img.get_pixel(x, y);
            let Rgba([r1, g1, b1, _a1]) = mask.get_pixel(x, y);

            let fr = apply(r0, r1);
            let fg = apply(g0, g1);
            let fb = apply(b0, b1);
            let fa = 255;

            out.put_pixel(x, y, Rgba([fr, fg, fb, fa]));
        }
    }

    out
}

// Calculate the light mask that, when applied to img will set all the colors darker than the input
// color to the input color. Colors brighter than the input will be set to transparent (alpha = 0).
fn img_light_mask(img: &DynamicImage, Rgb([r0, g0, b0]): Rgb<u8>) -> DynamicImage {
    let (w, h) = img.dimensions();
    let mut mask = img.clone();

    for x in 0..w {
        for y in 0..h {
            let mut invalid = false;
            let mut calc_light = |c0, c1| {
                let f = (c1 as f32) / (c0 as f32) * 255.0;
                if f > 255.0 {
                    invalid = true;
                    0
                } else {
                    f.round() as u8
                }
            };

            let Rgba([r1, g1, b1, _a1]) = img.get_pixel(x, y);

            let fr = calc_light(r0, r1);
            let fg = calc_light(g0, g1);
            let fb = calc_light(b0, b1);
            let fa = if invalid { 0 } else { 255 };

            mask.put_pixel(x, y, Rgba([fr, fg, fb, fa]));
        }
    }

    mask
}

fn combine_light_masks(img1: &DynamicImage, img2: &DynamicImage, thres: u8) -> DynamicImage {
    let mut out = img1.clone();
    let (w, h) = out.dimensions();

    for x in 0..w {
        for y in 0..h {
            let p1 = img1.get_pixel(x, y);
            let p2 = img2.get_pixel(x, y);
            let Rgba([r1, g1, b1, a1]) = p1;
            let Rgba([r2, g2, b2, a2]) = p2;

            // Pick the brightest of the two colors, but do not combine channels.
            // a1 and a2 are alpha, which should only be 0 or 255.
            let mut p = if a1 > a2 {
                p1
            } else if a2 > a1 {
                p2
            } else if a1 == 0 && a2 == 0 {
                // If both inputs are 0, set to 0
                Rgba([0, 0, 0, 255])
            } else {
                let sum1 = u16::from(r1) + u16::from(g1) + u16::from(b1);
                let sum2 = u16::from(r2) + u16::from(g2) + u16::from(b2);

                if sum1 >= sum2 {
                    p1
                } else {
                    p2
                }
            };

            let Rgba([rp, gp, bp, _ap]) = p;

            if rp < thres || gp < thres || bp < thres {
                p = Rgba([0, 0, 0, 255]);
            }

            out.put_pixel(x, y, p);
        }
    }

    out
}

fn repeated_rgb_dilation(img: &DynamicImage, max_steps: u32) -> DynamicImage {
    let mut dilated = img.clone();

    for _ in 0..max_steps {
        let mut any_black_pixels = false;
        dilated = rgb_dilation(&dilated, &mut any_black_pixels);
        if !any_black_pixels {
            break;
        }
    }

    dilated
}

fn rgb_dilation(img: &DynamicImage, any_black_pixels: &mut bool) -> DynamicImage {
    let mut out = img.clone();
    let (w, h) = out.dimensions();

    // TODO: this is a naive 3x3 median filter that could be optimized
    // imageproc::filters::median_filter does exactly what we need, but it doesn't work when the
    // input is a DynamicImage
    for x in 0..w {
        for y in 0..h {
            let p = img.get_pixel(x, y);
            let background = Rgba([0, 0, 0, 255]);
            if p == background {
                *any_black_pixels = true;
            }

            let median_3x3 = median_rgb_3x3(img, (x, y));
            out.put_pixel(x, y, median_3x3);
        }
    }

    out
}

// Calculate the median color of a 3x3 area around (x, y)
fn median_rgb_3x3(img: &DynamicImage, (x, y): (u32, u32)) -> Rgba<u8> {
    let mut colors = Vec::with_capacity(9);
    let (w, h) = img.dimensions();

    for x in x.saturating_sub(1)..std::cmp::min(x + 1, w) {
        for y in y.saturating_sub(1)..std::cmp::min(y + 1, h) {
            let p = img.get_pixel(x, y);
            colors.push(p);
        }
    }

    colors.sort_by_key(|p| {
        // Sort by luminance
        let Rgba([r, g, b, _a]) = p;

        u16::from(*r) + u16::from(*g) + u16::from(*b)
    });

    colors[colors.len() / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn odd_sized_map() {
        let img = DynamicImage::new_rgba8(129, 129);

        let _out_img = extract_unexplored_treasure_map(&img);
        //assert_eq!(out_img.dimensions(), (129, 129));
    }
}
