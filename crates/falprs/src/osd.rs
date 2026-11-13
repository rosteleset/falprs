//! On-screen display on saved screenshots: the current date and time in the upper left corner and
//! the video stream's title in the lower left corner, white with a black outline.
//!
//! The C++ service draws them with OpenCV's Hershey font, which covers Latin and Cyrillic letters
//! but not, for example, `№`; here the text is rendered with the bundled DejaVu Sans Bold, which
//! covers Russian titles fully. The text looks different but has about the same size and placement.

use std::fmt::Write;

use chrono::Local;
use image::RgbImage;
use swash::FontRef;
use swash::scale::{Render, ScaleContext, Source};
use swash::zeno::{Format, Vector};

static FONT_DATA: &[u8] = include_bytes!("../fonts/DejaVuSans-Bold.ttf");

fn font() -> FontRef<'static> {
    FontRef::from_index(FONT_DATA, 0).expect("the bundled font is valid")
}

/// Distance of the text from the frame edges, in pixels.
const MARGIN: f32 = 10.0;

/// Formats the current local time; an invalid format is printed as is.
fn now_text(format: &str) -> String {
    let mut text = String::new();
    if write!(text, "{}", Local::now().format(format)).is_err() {
        return format.to_string();
    }
    text
}

/// A rendered glyph's coverage placed in image coordinates.
struct GlyphImage {
    left: i32,
    top: i32,
    width: i32,
    coverage: Vec<u8>,
}

/// Renders `text` at `size` pixels per em with its baseline starting at (`x`, `baseline`).
fn rasterize(text: &str, size: f32, x: f32, baseline: f32) -> Vec<GlyphImage> {
    let font = font();
    let charmap = font.charmap();
    let metrics = font.glyph_metrics(&[]).scale(size);
    let mut context = ScaleContext::new();
    let mut scaler = context.builder(font).size(size).hint(false).build();
    let mut glyphs = Vec::new();
    let mut caret = x;
    for c in text.chars() {
        let id = charmap.map(c);
        let rendered = Render::new(&[Source::Outline])
            .format(Format::Alpha)
            .offset(Vector::new(caret.fract(), 0.0))
            .render(&mut scaler, id);
        if let Some(rendered) = rendered {
            let placement = rendered.placement;
            glyphs.push(GlyphImage {
                left: caret.floor() as i32 + placement.left,
                top: baseline.round() as i32 - placement.top,
                width: placement.width as i32,
                coverage: rendered.data,
            });
        }
        caret += metrics.advance_width(id);
    }
    glyphs
}

/// The font size at which digits are `height` pixels tall, the measure OpenCV uses for the scale.
fn size_for_digit_height(height: f32) -> f32 {
    const UNIT: f32 = 100.0;
    let digit_height = rasterize("0", UNIT, 0.0, 0.0)
        .first()
        .map_or(0.0, |glyph| (glyph.coverage.len() as i32 / glyph.width.max(1)) as f32);
    let digit_height = if digit_height > 0.0 { digit_height } else { 73.0 };
    (UNIT * height / digit_height).max(1.0)
}

/// Blends `color` into the image by the glyphs' coverage, shifted by `offset`.
fn blend(image: &mut RgbImage, glyphs: &[GlyphImage], offset: (i32, i32), color: [u8; 3]) {
    let (width, height) = (image.width() as i32, image.height() as i32);
    for glyph in glyphs {
        for (i, &coverage) in glyph.coverage.iter().enumerate() {
            let i = i as i32;
            let px = glyph.left + i % glyph.width + offset.0;
            let py = glyph.top + i / glyph.width + offset.1;
            if coverage == 0 || px < 0 || py < 0 || px >= width || py >= height {
                continue;
            }
            let a = f32::from(coverage) / 255.0;
            let pixel = image.get_pixel_mut(px as u32, py as u32);
            for (channel, target) in pixel.0.iter_mut().zip(color) {
                *channel = (f32::from(*channel) * (1.0 - a) + f32::from(target) * a).round() as u8;
            }
        }
    }
}

/// Draws white text with a black outline, like the C++ service's two passes of `putText`.
fn draw_outlined(image: &mut RgbImage, text: &str, size: f32, x: f32, baseline: f32) {
    let glyphs = rasterize(text, size, x, baseline);
    let radius = (size / 24.0).round().max(1.0) as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                blend(image, &glyphs, (dx, dy), [0, 0, 0]);
            }
        }
    }
    blend(image, &glyphs, (0, 0), [255, 255, 255]);
}

/// The frame with the date and `title` drawn, or `None` for an empty title (no OSD).
pub fn draw(frame: &RgbImage, title: &str, date_format: &str, title_height_ratio: f32) -> Option<RgbImage> {
    if title.is_empty() {
        return None;
    }
    let text_height = (title_height_ratio * frame.height() as f32).max(1.0);
    let size = size_for_digit_height(text_height);
    let mut image = frame.clone();
    draw_outlined(&mut image, &now_text(date_format), size, MARGIN, MARGIN + text_height);
    draw_outlined(&mut image, title, size, MARGIN, frame.height() as f32 - MARGIN);
    Some(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    /// Rows of the image containing white text pixels.
    fn white_rows(image: &RgbImage) -> Vec<u32> {
        let mut rows: Vec<u32> = image.enumerate_pixels().filter(|(_, _, p)| p.0 == [255, 255, 255]).map(|(_, y, _)| y).collect();
        rows.dedup();
        rows
    }

    #[test]
    fn no_title_no_osd() {
        assert!(draw(&RgbImage::new(100, 100), "", "%Y", 0.05).is_none());
    }

    #[test]
    fn draws_date_and_cyrillic_title() {
        let frame = RgbImage::from_pixel(640, 480, Rgb([90, 120, 150]));
        let image = draw(&frame, "Подъезд №1", "%Y-%m-%d %H:%M:%S", 0.05).unwrap();
        let rows = white_rows(&image);
        // Digits 24 pixels tall at the top, the title just above the bottom margin
        assert!(rows.iter().any(|&y| y < 50) && rows.iter().all(|&y| y >= 8));
        assert!(rows.iter().any(|&y| y > 430 && y < 470));
        // The Cyrillic glyphs exist in the font: the title is as wide as its Latin transliteration or more
        let title_pixels = image.enumerate_pixels().filter(|(_, y, p)| *y > 400 && p.0 == [255, 255, 255]).count();
        assert!(title_pixels > 300, "{title_pixels}");
        let charmap = font().charmap();
        for c in "Подъезд №".chars() {
            assert_ne!(charmap.map(c), 0, "no glyph for {c}");
        }
    }

    #[test]
    fn invalid_date_format_is_printed_as_is() {
        assert_eq!(now_text("%Q"), "%Q");
    }
}
