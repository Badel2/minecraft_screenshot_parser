# minecraft_screenshot_parser

Image processing utilities to obtain information from Minecraft screenshots.

To be used in [slime_seed_finder](https://github.com/Badel2/slime_seed_finder).

## Status

Under development.

Current features:

* Detect and remove crosshair

Planned features:

* Detect and parse maps

### Remove crosshair

```sh
cargo run --release --example remove_crosshair -- screenshot.png
```

This will create a file `screenshot_nc.png` with the crosshair removed:

![5_c](https://raw.githubusercontent.com/Badel2/minecraft_screenshot_parser/master/blob/5.png)

![5_nc](https://raw.githubusercontent.com/Badel2/minecraft_screenshot_parser/master/blob/5_nc.png)


