//! Remove the crosshair from all the png images supplied as arguments.
use minecraft_screenshot_parser::crosshair::remove_crosshair;

fn main() {
    for img_path in std::env::args().skip(1) {
        let mut img = image::open(&img_path).unwrap();
        println!("Processing {}", img_path);

        let crosshair = remove_crosshair(&mut img);
        if let Some(crosshair) = crosshair {
            println!("Found crosshair: {:?}", crosshair);
            let out_path = img_path
                .strip_suffix(".png")
                .expect("image path should end with .png");
            let out_path = format!("{}{}", out_path.to_owned(), "_nc.png");
            img.save(&out_path).unwrap();
            println!("Image with crosshair removed saved as {}", out_path);
        } else {
            println!("No crosshair found");
        }
    }
}
