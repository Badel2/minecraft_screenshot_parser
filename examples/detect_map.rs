//! Extract the map from all the png images supplied as arguments.
use minecraft_screenshot_parser::crosshair::remove_crosshair;
use minecraft_screenshot_parser::map::detect_map;
use minecraft_screenshot_parser::map_color_correct;

fn main() {
    for img_path in std::env::args().skip(1) {
        let mut img = image::open(&img_path).unwrap();
        println!("Processing {}", img_path);

        let crosshair = remove_crosshair(&mut img);

        match crosshair {
            Some(_) => println!("Crosshair removed"),
            None => println!("No crosshair found"),
        }

        let out_path = img_path
            .strip_suffix(".png")
            .expect("image path should end with .png");

        let x = detect_map(&img).expect("map not found");
        //println!("{:?}", x);
        let cropped_img = &x.cropped_img;
        let cropped_scaled_img = &x.cropped_scaled_img;
        let img_with_bounding_box = &x.img_with_bounding_box;

        let i = "1_detected_map";
        let out_path1 = format!("{}{}{}{}", out_path.to_owned(), "_", i, ".png");
        img_with_bounding_box.save(&out_path1).unwrap();

        let i = "2_cropped_map";
        let out_path1 = format!("{}{}{}{}", out_path.to_owned(), "_", i, ".png");
        cropped_img.save(&out_path1).unwrap();

        let i = "3_cropped_scaled_map";
        let out_path1 = format!("{}{}{}{}", out_path.to_owned(), "_", i, ".png");
        cropped_scaled_img.save(&out_path1).unwrap();

        let palette_image = map_color_correct::extract_unexplored_treasure_map(cropped_scaled_img);
        let i = "4_palette";
        let out_path1 = format!("{}{}{}{}", out_path.to_owned(), "_", i, ".png");
        palette_image.save(&out_path1).unwrap();
    }
}
