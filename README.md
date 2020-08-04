# minecraft_screenshot_parser

Image processing utilities to obtain information from Minecraft screenshots.

To be used in [slime_seed_finder](https://github.com/Badel2/slime_seed_finder).

## Status

Under development.
Currently only lossless PNG screenshots are supported, any kind of JPG noise will make the
algorithms fail.

Current features:

* Detect and remove crosshair
* Detect and parse treasure maps

See the `examples/` folder for usage examples.

Planned features:

* Detect and parse terrain maps

### Remove crosshair

```sh
cargo run --release --example remove_crosshair -- screenshot.png
```

This will create a file `screenshot_nc.png` with the crosshair removed:

![5_c](https://raw.githubusercontent.com/Badel2/minecraft_screenshot_parser/master/blob/5.png)

![5_nc](https://raw.githubusercontent.com/Badel2/minecraft_screenshot_parser/master/blob/5_nc.png)

### Detect treasure map

```sh
cargo run --release --example detect_map -- screenshot.png
```

![Example image with markers](https://raw.githubusercontent.com/Badel2/minecraft_screenshot_parser/master/blob/5_map_marks.png)

![Example image extracted treasure map](https://raw.githubusercontent.com/Badel2/minecraft_screenshot_parser/master/blob/5_map_cropped.png)

