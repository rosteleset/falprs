//! Image decoding and model input preparation, reproducing the OpenCV operations of the C++
//! service closely enough for the models to produce the same detections.

use std::io::Cursor;

use anyhow::{Context, Result};
use image::{DynamicImage, ImageDecoder, ImageReader, Rgb, RgbImage};

/// Decodes an image, applying its EXIF orientation like OpenCV's `imdecode` with `IMREAD_COLOR`.
pub fn decode(data: &[u8]) -> Result<RgbImage> {
    let mut decoder = ImageReader::new(Cursor::new(data))
        .with_guessed_format()?
        .into_decoder()
        .context("unsupported image format")?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder).context("can't decode image")?;
    image.apply_orientation(orientation);
    Ok(image.into_rgb8())
}

/// Fixed-point scale of OpenCV's resize interpolation coefficients (`INTER_RESIZE_COEF_SCALE`).
const COEF_SCALE: f32 = 2048.0;

/// Source index and fixed-point weights of one destination coordinate.
type Tap = (usize, i32, i32);

fn make_tap(src_len: u32, s: i64, frac: f32) -> Tap {
    let (s, frac) = if s < 0 {
        (0, 0.0)
    } else if s >= src_len as i64 - 1 {
        (src_len as i64 - 1, 0.0)
    } else {
        (s, frac)
    };
    let w0 = ((1.0 - frac) * COEF_SCALE).round_ties_even() as i32;
    (s as usize, w0, COEF_SCALE as i32 - w0)
}

/// Taps of OpenCV's `INTER_LINEAR`: half-pixel centres, no antialiasing, edges clamped.
fn linear_taps(src_len: u32, dst_len: u32) -> Vec<Tap> {
    let scale = src_len as f64 / dst_len as f64;
    (0..dst_len)
        .map(|d| {
            let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let s = f.floor();
            make_tap(src_len, s as i64, f - s)
        })
        .collect()
}

/// Taps of OpenCV's `INTER_AREA` when the image is enlarged along at least one axis.
fn area_upscale_taps(src_len: u32, dst_len: u32) -> Vec<Tap> {
    let scale = src_len as f64 / dst_len as f64;
    let inv_scale = dst_len as f64 / src_len as f64;
    (0..dst_len)
        .map(|d| {
            let s = (d as f64 * scale).floor() as i64;
            let f = ((d + 1) as f64 - (s + 1) as f64 * inv_scale) as f32;
            let f = if f <= 0.0 { 0.0 } else { f - f.floor() };
            make_tap(src_len, s, f)
        })
        .collect()
}

/// Two-pass fixed-point interpolation with the given taps, rounding like OpenCV.
fn resize_with_taps(src: &RgbImage, width: u32, height: u32, xs: &[Tap], ys: &[Tap]) -> RgbImage {
    let (src_w, src_h) = src.dimensions();
    let src_raw = src.as_raw();
    let row_len = src_w as usize * 3;

    // Horizontal pass of one source row into fixed-point intermediate values
    let horizontal = |y: usize| -> Vec<i32> {
        let row = &src_raw[y * row_len..(y + 1) * row_len];
        let mut out = Vec::with_capacity(width as usize * 3);
        for &(sx, w0, w1) in xs {
            let next = (sx + 1).min(src_w as usize - 1);
            for c in 0..3 {
                out.push(row[sx * 3 + c] as i32 * w0 + row[next * 3 + c] as i32 * w1);
            }
        }
        out
    };

    let mut dst = RgbImage::new(width, height);
    let dst_raw: &mut [u8] = &mut dst;
    let mut cached: Option<(usize, Vec<i32>)> = None;
    for (dy, &(sy, w0, w1)) in ys.iter().enumerate() {
        let row0 = match cached.take() {
            Some((y, row)) if y == sy => row,
            _ => horizontal(sy),
        };
        let row1 = horizontal((sy + 1).min(src_h as usize - 1));
        let out = &mut dst_raw[dy * width as usize * 3..(dy + 1) * width as usize * 3];
        for (i, value) in out.iter_mut().enumerate() {
            // OpenCV's vectorised vertical pass: products of 16-bit values keep their high halves
            let sum = ((w0 * (row0[i] >> 4)) >> 16) + ((w1 * (row1[i] >> 4)) >> 16) + 2;
            *value = (sum >> 2).clamp(0, 255) as u8;
        }
        cached = Some((sy + 1, row1));
    }
    dst
}

/// Bilinear resize with OpenCV `INTER_LINEAR` semantics and rounding.
pub fn resize_linear(src: &RgbImage, width: u32, height: u32) -> RgbImage {
    let xs = linear_taps(src.width(), width);
    let ys = linear_taps(src.height(), height);
    resize_with_taps(src, width, height, &xs, &ys)
}

/// Source index and weight of one source cell contributing to a destination cell.
struct AreaTap {
    dst: usize,
    src: usize,
    weight: f32,
}

/// OpenCV's `computeResizeAreaTab`.
fn area_taps(src_len: u32, dst_len: u32, scale: f64) -> Vec<AreaTap> {
    let mut taps = Vec::new();
    for d in 0..dst_len as usize {
        let fs1 = d as f64 * scale;
        let fs2 = fs1 + scale;
        let cell = scale.min(src_len as f64 - fs1);
        let s2 = (fs2.floor() as i64).min(src_len as i64 - 1);
        let s1 = (fs1.ceil() as i64).min(s2);
        if s1 as f64 - fs1 > 1e-3 {
            taps.push(AreaTap { dst: d, src: (s1 - 1) as usize, weight: ((s1 as f64 - fs1) / cell) as f32 });
        }
        for s in s1..s2 {
            taps.push(AreaTap { dst: d, src: s as usize, weight: (1.0 / cell) as f32 });
        }
        if fs2 - s2 as f64 > 1e-3 {
            taps.push(AreaTap { dst: d, src: s2 as usize, weight: ((fs2 - s2 as f64).min(1.0).min(cell) / cell) as f32 });
        }
    }
    taps
}

/// Resize with OpenCV `INTER_AREA` semantics: pixel area averaging when shrinking,
/// a variant of bilinear interpolation when enlarging.
pub fn resize_area(src: &RgbImage, width: u32, height: u32) -> RgbImage {
    let (src_w, src_h) = src.dimensions();
    let scale_x = src_w as f64 / width as f64;
    let scale_y = src_h as f64 / height as f64;
    if scale_x < 1.0 || scale_y < 1.0 {
        let xs = area_upscale_taps(src_w, width);
        let ys = area_upscale_taps(src_h, height);
        return resize_with_taps(src, width, height, &xs, &ys);
    }

    let src_raw = src.as_raw();
    let mut dst = RgbImage::new(width, height);
    let (ix, iy) = (scale_x.round(), scale_y.round());
    if (scale_x - ix).abs() < f64::EPSILON && (scale_y - iy).abs() < f64::EPSILON {
        // Whole-pixel cells: plain averages
        let (ix, iy) = (ix as usize, iy as usize);
        let area = (ix * iy) as i32;
        let inv_area = 1.0 / area as f32;
        for (x, y, pixel) in dst.enumerate_pixels_mut() {
            for c in 0..3 {
                let mut sum = 0;
                for sy in y as usize * iy..(y as usize + 1) * iy {
                    for sx in x as usize * ix..(x as usize + 1) * ix {
                        sum += src_raw[(sy * src_w as usize + sx) * 3 + c] as i32;
                    }
                }
                pixel[c] = if area == 4 {
                    ((sum + 2) >> 2) as u8
                } else {
                    (sum as f32 * inv_area).round_ties_even().clamp(0.0, 255.0) as u8
                };
            }
        }
        return dst;
    }

    let xs = area_taps(src_w, width, scale_x);
    let ys = area_taps(src_h, height, scale_y);
    let row_len = width as usize * 3;
    let dst_raw: &mut [u8] = &mut dst;
    let mut sum = vec![0.0f32; row_len];
    let mut buf = vec![0.0f32; row_len];
    let mut previous: Option<usize> = None;
    let write = |dy: usize, sum: &[f32], dst_raw: &mut [u8]| {
        for (value, s) in dst_raw[dy * row_len..(dy + 1) * row_len].iter_mut().zip(sum) {
            *value = s.round_ties_even().clamp(0.0, 255.0) as u8;
        }
    };
    for tap_y in &ys {
        buf.iter_mut().for_each(|b| *b = 0.0);
        let row = &src_raw[tap_y.src * src_w as usize * 3..(tap_y.src + 1) * src_w as usize * 3];
        for tap_x in &xs {
            for c in 0..3 {
                buf[tap_x.dst * 3 + c] += row[tap_x.src * 3 + c] as f32 * tap_x.weight;
            }
        }
        if previous == Some(tap_y.dst) {
            for (s, b) in sum.iter_mut().zip(&buf) {
                *s += b * tap_y.weight;
            }
        } else {
            if let Some(dy) = previous {
                write(dy, &sum, dst_raw);
            }
            for (s, b) in sum.iter_mut().zip(&buf) {
                *s = b * tap_y.weight;
            }
            previous = Some(tap_y.dst);
        }
    }
    if let Some(dy) = previous {
        write(dy, &sum, dst_raw);
    }
    dst
}

/// The part of `image` inside `[x1, y1, x2, y2)`, like OpenCV's ROI of `cv::Rect(Point, Point)`.
pub fn crop(image: &RgbImage, x1: i32, y1: i32, x2: i32, y2: i32) -> Result<RgbImage> {
    let (left, right) = (x1.min(x2), x1.max(x2));
    let (top, bottom) = (y1.min(y2), y1.max(y2));
    if left < 0 || top < 0 || right as u32 > image.width() || bottom as u32 > image.height() || left == right || top == bottom {
        anyhow::bail!("invalid region [{x1}, {y1}, {x2}, {y2}] of a {}x{} image", image.width(), image.height());
    }
    Ok(image::imageops::crop_imm(image, left as u32, top as u32, (right - left) as u32, (bottom - top) as u32).to_image())
}

/// Adds a constant border, like OpenCV's `copyMakeBorder` with `BORDER_CONSTANT`.
pub fn pad(src: &RgbImage, top: u32, bottom: u32, left: u32, right: u32, value: u8) -> RgbImage {
    let (w, h) = src.dimensions();
    let mut dst = RgbImage::from_pixel(w + left + right, h + top + bottom, Rgb([value; 3]));
    image::imageops::replace(&mut dst, src, left as i64, top as i64);
    dst
}

/// Packs an RGB image into an NCHW float tensor with `(pixel - mean) * scale`, like OpenCV's
/// `blobFromImage` with `swapRB` on a BGR image.
pub fn to_tensor(image: &RgbImage, scale: f32, mean: f32) -> Vec<f32> {
    let plane = (image.width() * image.height()) as usize;
    let mut tensor = vec![0.0; plane * 3];
    for (i, pixel) in image.pixels().enumerate() {
        for c in 0..3 {
            tensor[c * plane + i] = (pixel[c] as f32 - mean) * scale;
        }
    }
    tensor
}

/// A model input prepared by letterboxing.
pub struct Letterbox {
    pub tensor: Vec<f32>,
    pub width: u32,
    pub height: u32,
    /// Padding added on the left and on the top, in model input pixels.
    pub padding: (f32, f32),
    /// Model input size divided by image size.
    pub ratio: f32,
}

/// Ultralytics `LetterBox` with `auto=True` for YOLO OBB models: scale to fit `target_size`,
/// then pad only up to a multiple of the model stride.
pub fn letterbox_yolo_obb(image: &RgbImage, target_size: u32) -> Letterbox {
    const STRIDE: i64 = 32;
    const PADDING_VALUE: u8 = 114;

    let (w, h) = image.dimensions();
    let ratio = (target_size as f64 / h as f64).min(target_size as f64 / w as f64);
    let new_w = (w as f64 * ratio).round() as u32;
    let new_h = (h as f64 * ratio).round() as u32;
    let dw = (target_size as i64 - new_w as i64) % STRIDE;
    let dh = (target_size as i64 - new_h as i64) % STRIDE;
    let padding = (dw as f32 / 2.0, dh as f32 / 2.0);

    let resized = if (new_w, new_h) == (w, h) {
        image.clone()
    } else {
        resize_linear(image, new_w, new_h)
    };
    let top = (padding.1 - 0.1).round() as u32;
    let bottom = (padding.1 + 0.1).round() as u32;
    let left = (padding.0 - 0.1).round() as u32;
    let right = (padding.0 + 0.1).round() as u32;
    let letterboxed = pad(&resized, top, bottom, left, right, PADDING_VALUE);

    Letterbox {
        tensor: to_tensor(&letterboxed, 1.0 / 255.0, 0.0),
        width: letterboxed.width(),
        height: letterboxed.height(),
        padding,
        ratio: ratio as f32,
    }
}

/// A centred letterbox with a constant 114 border, as the C++ service prepares inputs for
/// YOLO detectors and the plate classifier. Returns the image, its offset and its scale.
fn letterbox_centred(image: &RgbImage, width: u32, height: u32) -> (RgbImage, (f32, f32), f64) {
    const PADDING_VALUE: u8 = 114;
    let scale = (width as f64 / image.width() as f64).min(height as f64 / image.height() as f64);
    let new_w = (scale * image.width() as f64).round() as u32;
    let new_h = (scale * image.height() as f64).round() as u32;
    let shift = ((width - new_w) as f32 / 2.0, (height - new_h) as f32 / 2.0);
    let mut out = RgbImage::from_pixel(width, height, Rgb([PADDING_VALUE; 3]));
    image::imageops::replace(&mut out, &resize_linear(image, new_w, new_h), shift.0 as i64, shift.1 as i64);
    (out, shift, scale)
}

/// A YOLO detector input: the letterboxed image scaled to `[0, 1]`.
pub struct YoloInput {
    pub tensor: Vec<f32>,
    /// Offset of the image inside the input.
    pub shift: (f32, f32),
    /// Input size divided by image size.
    pub scale: f64,
}

pub fn prepare_yolo(image: &RgbImage, width: u32, height: u32) -> YoloInput {
    let (out, shift, scale) = letterbox_centred(image, width, height);
    YoloInput {
        tensor: to_tensor(&out, 1.0 / 255.0, 0.0),
        shift,
        scale,
    }
}

/// Plate classifier input: letterboxed and scaled to `[-1, 1]`.
pub fn prepare_lpc(image: &RgbImage, width: u32, height: u32) -> Vec<f32> {
    let (out, _, _) = letterbox_centred(image, width, height);
    to_tensor(&out, 1.0 / 127.5, 127.5)
}

/// Special vehicle classifier input: area-resized and scaled to `[-1, 1]`.
pub fn prepare_vc(image: &RgbImage, width: u32, height: u32) -> Vec<f32> {
    to_tensor(&resize_area(image, width, height), 1.0 / 127.5, 127.5)
}

/// Solves `a * x = b` by Gaussian elimination with partial pivoting.
fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let pivot = (col..N).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < f64::EPSILON {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..N {
            let factor = a[row][col] / a[col][col];
            for k in col..N {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; N];
    for row in (0..N).rev() {
        let tail: f64 = (row + 1..N).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    Some(x)
}

/// A 3x3 projective transform, row-major.
pub type Homography = [f64; 9];

/// OpenCV's `getPerspectiveTransform`: the homography mapping four `src` points onto `dst`.
pub fn perspective_transform(src: [[f32; 2]; 4], dst: [[f32; 2]; 4]) -> Option<Homography> {
    let mut a = [[0.0; 8]; 8];
    let mut b = [0.0; 8];
    for i in 0..4 {
        let (x, y) = (src[i][0] as f64, src[i][1] as f64);
        let (u, v) = (dst[i][0] as f64, dst[i][1] as f64);
        a[i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u];
        a[i + 4] = [0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v];
        b[i] = u;
        b[i + 4] = v;
    }
    let m = solve(a, b)?;
    Some([m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], 1.0])
}

fn invert(m: &Homography) -> Option<Homography> {
    let mut inverse = [0.0; 9];
    for col in 0..3 {
        let mut a = [[0.0; 3]; 3];
        for (row, a_row) in a.iter_mut().enumerate() {
            a_row.copy_from_slice(&m[row * 3..row * 3 + 3]);
        }
        let mut e = [0.0; 3];
        e[col] = 1.0;
        let x = solve(a, e)?;
        for row in 0..3 {
            inverse[row * 3 + col] = x[row];
        }
    }
    Some(inverse)
}

/// OpenCV's `warpPerspective` with `INTER_LINEAR` and a black constant border: source positions
/// are quantised to 1/32 pixel and blended with 15-bit fixed-point weights.
pub fn warp_perspective(src: &RgbImage, m: &Homography, width: u32, height: u32) -> Option<RgbImage> {
    const INTER_BITS: i32 = 5;
    const INTER_TAB_SIZE: f64 = (1 << INTER_BITS) as f64;
    const COEF_BITS: i32 = 15;

    let inv = invert(m)?;
    let (src_w, src_h) = (src.width() as i64, src.height() as i64);
    let pixel = |x: i64, y: i64, c: usize| -> i32 {
        if x < 0 || y < 0 || x >= src_w || y >= src_h {
            0
        } else {
            src.get_pixel(x as u32, y as u32)[c] as i32
        }
    };
    let mut dst = RgbImage::new(width, height);
    for y in 0..height {
        let (x0, y0, w0) = (
            inv[1] * y as f64 + inv[2],
            inv[4] * y as f64 + inv[5],
            inv[7] * y as f64 + inv[8],
        );
        for x in 0..width {
            let w = w0 + inv[6] * x as f64;
            let w = if w != 0.0 { INTER_TAB_SIZE / w } else { 0.0 };
            let fx = ((x0 + inv[0] * x as f64) * w).clamp(i32::MIN as f64, i32::MAX as f64);
            let fy = ((y0 + inv[3] * x as f64) * w).clamp(i32::MIN as f64, i32::MAX as f64);
            let (qx, qy) = (fx.round_ties_even() as i64, fy.round_ties_even() as i64);
            let (sx, sy) = (qx >> INTER_BITS, qy >> INTER_BITS);
            let (ax, ay) = ((qx & 31) as i32, (qy & 31) as i32);
            // Products of (1 - t, t) at 1/32 steps are exact in 15-bit fixed point
            let weights = [(32 - ax) * (32 - ay) * 32, ax * (32 - ay) * 32, (32 - ax) * ay * 32, ax * ay * 32];
            let out = dst.get_pixel_mut(x, y);
            for c in 0..3 {
                let sum = pixel(sx, sy, c) * weights[0]
                    + pixel(sx + 1, sy, c) * weights[1]
                    + pixel(sx, sy + 1, c) * weights[2]
                    + pixel(sx + 1, sy + 1, c) * weights[3];
                out[c] = ((sum + (1 << (COEF_BITS - 1))) >> COEF_BITS).clamp(0, 255) as u8;
            }
        }
    }
    Some(dst)
}

/// Blends four neighbours with OpenCV's 15-bit fixed-point bilinear weights for a source position
/// quantised to 1/32 pixel; pixels outside the image are black.
fn remap_linear(src: &RgbImage, qx: i64, qy: i64) -> [u8; 3] {
    const INTER_BITS: i64 = 5;
    const COEF_BITS: i32 = 15;
    let (w, h) = (src.width() as i64, src.height() as i64);
    let (sx, sy) = (qx >> INTER_BITS, qy >> INTER_BITS);
    let (ax, ay) = ((qx & 31) as i32, (qy & 31) as i32);
    // Products of (1 - t, t) at 1/32 steps are exact in 15-bit fixed point
    let weights = [(32 - ax) * (32 - ay) * 32, ax * (32 - ay) * 32, (32 - ax) * ay * 32, ax * ay * 32];
    let pixel = |x: i64, y: i64, c: usize| -> i32 {
        if x < 0 || y < 0 || x >= w || y >= h { 0 } else { src.get_pixel(x as u32, y as u32)[c] as i32 }
    };
    let mut out = [0; 3];
    for (c, value) in out.iter_mut().enumerate() {
        let sum = pixel(sx, sy, c) * weights[0]
            + pixel(sx + 1, sy, c) * weights[1]
            + pixel(sx, sy + 1, c) * weights[2]
            + pixel(sx + 1, sy + 1, c) * weights[3];
        *value = ((sum + (1 << (COEF_BITS - 1))) >> COEF_BITS).clamp(0, 255) as u8;
    }
    out
}

/// A 2x3 affine transform, row-major.
pub type Affine = [f64; 6];

/// OpenCV's `warpAffine` with `INTER_LINEAR` and a black constant border.
pub fn warp_affine(src: &RgbImage, m: &Affine, width: u32, height: u32) -> RgbImage {
    const AB_BITS: i32 = 10;
    const AB_SCALE: f64 = (1 << AB_BITS) as f64;
    const ROUND_DELTA: i64 = (1 << AB_BITS) / 32 / 2;
    // Inverse transform, as OpenCV's invertAffineTransform computes it
    let det = m[0] * m[4] - m[1] * m[3];
    let d = if det != 0.0 { 1.0 / det } else { 0.0 };
    let (a11, a22, a12, a21) = (m[4] * d, m[0] * d, -m[1] * d, -m[3] * d);
    let b1 = -a11 * m[2] - a12 * m[5];
    let b2 = -a21 * m[2] - a22 * m[5];
    let fixed = |v: f64| v.round_ties_even() as i64;
    let adelta: Vec<i64> = (0..width).map(|x| fixed(a11 * x as f64 * AB_SCALE)).collect();
    let bdelta: Vec<i64> = (0..width).map(|x| fixed(a21 * x as f64 * AB_SCALE)).collect();
    let mut dst = RgbImage::new(width, height);
    for y in 0..height {
        let x0 = fixed((a12 * y as f64 + b1) * AB_SCALE) + ROUND_DELTA;
        let y0 = fixed((a22 * y as f64 + b2) * AB_SCALE) + ROUND_DELTA;
        for x in 0..width {
            let qx = (x0 + adelta[x as usize]) >> (AB_BITS - 5);
            let qy = (y0 + bdelta[x as usize]) >> (AB_BITS - 5);
            dst.put_pixel(x, y, Rgb(remap_linear(src, qx, qy)));
        }
    }
    dst
}

/// The similarity (rotation, uniform scale, translation) best mapping `from` onto `to` in the
/// least-squares sense.
fn similarity_least_squares(from: &[[f64; 2]], to: &[[f64; 2]]) -> Option<Affine> {
    let n = from.len() as f64;
    let mean = |points: &[[f64; 2]]| {
        let (sx, sy) = points.iter().fold((0.0, 0.0), |(x, y), p| (x + p[0], y + p[1]));
        [sx / n, sy / n]
    };
    let (mf, mt) = (mean(from), mean(to));
    let (mut num_a, mut num_b, mut den) = (0.0, 0.0, 0.0);
    for (p, q) in from.iter().zip(to) {
        let (px, py, qx, qy) = (p[0] - mf[0], p[1] - mf[1], q[0] - mt[0], q[1] - mt[1]);
        num_a += px * qx + py * qy;
        num_b += px * qy - py * qx;
        den += px * px + py * py;
    }
    if den < f64::EPSILON {
        return None;
    }
    let (a, b) = (num_a / den, num_b / den);
    Some([a, -b, mt[0] - a * mf[0] + b * mf[1], b, a, mt[1] - b * mf[0] - a * mf[1]])
}

fn squared_error(m: &Affine, p: [f64; 2], q: [f64; 2]) -> f64 {
    let x = m[0] * p[0] + m[1] * p[1] + m[2] - q[0];
    let y = m[3] * p[0] + m[4] * p[1] + m[5] - q[1];
    x * x + y * y
}

/// OpenCV's `estimateAffinePartial2D` with `LMEDS`: the two-point similarity with the least median
/// squared error over all pairs (OpenCV samples pairs at random), its inliers by OpenCV's
/// threshold, then the least-squares similarity of the inliers (where OpenCV's
/// Levenberg-Marquardt refinement converges).
pub fn estimate_affine_partial_lmeds(from: &[[f32; 2]], to: &[[f32; 2]]) -> Option<Affine> {
    let from: Vec<[f64; 2]> = from.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    let to: Vec<[f64; 2]> = to.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    let n = from.len();
    if n < 2 || n != to.len() {
        return None;
    }
    let mut best: Option<(f64, Affine)> = None;
    for i in 0..n {
        for j in i + 1..n {
            let Some(model) = similarity_least_squares(&[from[i], from[j]], &[to[i], to[j]]) else {
                continue;
            };
            let mut errors: Vec<f64> = from.iter().zip(&to).map(|(p, q)| squared_error(&model, *p, *q)).collect();
            errors.sort_by(f64::total_cmp);
            let median = errors[n / 2];
            if best.is_none_or(|(m, _)| median < m) {
                best = Some((median, model));
            }
        }
    }
    let (median, model) = best?;
    if n == 2 {
        return Some(model);
    }
    let sigma = 2.5 * 1.4826 * (1.0 + 5.0 / (n - 2) as f64) * median.sqrt();
    let (inliers_from, inliers_to): (Vec<_>, Vec<_>) = from
        .iter()
        .zip(&to)
        .filter(|(p, q)| squared_error(&model, **p, **q) <= sigma * sigma)
        .map(|(p, q)| (*p, *q))
        .unzip();
    if inliers_from.len() < 2 {
        return Some(model);
    }
    similarity_least_squares(&inliers_from, &inliers_to)
}

/// SCRFD face detector input: scaled to fit at the top left of a black canvas, `(pixel - 127.5) / 128`.
/// Returns the tensor and the scale of the image inside it.
pub fn prepare_scrfd(image: &RgbImage, width: u32, height: u32) -> (Vec<f32>, f32) {
    let r_w = width as f64 / image.width() as f64;
    let r_h = height as f64 / image.height() as f64;
    let (w, h) = if r_h > r_w {
        (width, (r_w * image.height() as f64) as u32)
    } else {
        ((r_h * image.width() as f64) as u32, height)
    };
    let scale = h as f32 / image.height() as f32;
    let mut out = RgbImage::new(width, height);
    image::imageops::replace(&mut out, &resize_linear(image, w, h), 0, 0);
    (to_tensor(&out, 1.0 / 128.0, 127.5), scale)
}

/// ArcFace input: resized to the model input, `(pixel - 127.5) / 127.5`.
pub fn prepare_arcface(face: &RgbImage, width: u32, height: u32) -> Vec<f32> {
    let face = if face.dimensions() == (width, height) { face.clone() } else { resize_linear(face, width, height) };
    to_tensor(&face, 1.0 / 127.5, 127.5)
}

/// Face class (GENet) input: area-resized to the model input, ImageNet-normalized RGB planes.
pub fn prepare_genet(face: &RgbImage, width: u32, height: u32) -> Vec<f32> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let face = if face.dimensions() == (width, height) { face.clone() } else { resize_area(face, width, height) };
    let plane = (width * height) as usize;
    let mut tensor = vec![0.0; plane * 3];
    for (i, pixel) in face.pixels().enumerate() {
        for c in 0..3 {
            tensor[c * plane + i] = (pixel[c] as f32 * (1.0 / 255.0) - MEAN[c]) * (1.0 / STD[c]);
        }
    }
    tensor
}

/// Focus measure of the C++ service (Pech2000 "LAPV"): the variance of the 3x3 Laplacian of the
/// blue channel (OpenCV's first channel), leaving out a 3-pixel border.
pub fn variance_of_laplacian(image: &RgbImage) -> f64 {
    const MARGIN: u32 = 3;
    let (width, height) = image.dimensions();
    if width <= 2 * MARGIN || height <= 2 * MARGIN {
        return 0.0;
    }
    let blue = |x: u32, y: u32| f64::from(image.get_pixel(x, y)[2]);
    let (mut sum, mut sqsum) = (0.0, 0.0);
    for y in MARGIN..height - MARGIN {
        for x in MARGIN..width - MARGIN {
            let v = blue(x, y - 1) + blue(x - 1, y) + blue(x + 1, y) + blue(x, y + 1) - 4.0 * blue(x, y);
            sum += v;
            sqsum += v * v;
        }
    }
    let n = f64::from((width - 2 * MARGIN) * (height - 2 * MARGIN));
    let mean = sum / n;
    (sqsum / n - mean * mean).max(0.0)
}

/// Encodes an image as JPEG with OpenCV's default quality.
pub fn encode_jpeg(image: &RgbImage) -> Result<Vec<u8>> {
    let mut encoded = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, 95).encode_image(image)?;
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_resize_keeps_pixels() {
        let image = RgbImage::from_fn(5, 4, |x, y| Rgb([(x * 40) as u8, (y * 60) as u8, 7]));
        assert_eq!(resize_linear(&image, 5, 4), image);
    }

    #[test]
    fn upscale_interpolates_between_pixels() {
        let image = RgbImage::from_fn(2, 1, |x, _| Rgb([if x == 0 { 0 } else { 200 }; 3]));
        let resized = resize_linear(&image, 4, 1);
        let row: Vec<u8> = resized.pixels().map(|p| p[0]).collect();
        // OpenCV: centres at -0.25 (clamped), 0.25, 0.75, 1.25 (clamped)
        assert_eq!(row, vec![0, 50, 150, 200]);
    }

    #[test]
    fn area_resize_averages_cells() {
        let image = RgbImage::from_fn(4, 2, |x, _| Rgb([(x * 10) as u8; 3]));
        let row: Vec<u8> = resize_area(&image, 2, 1).pixels().map(|p| p[0]).collect();
        assert_eq!(row, vec![5, 25]);
        let row: Vec<u8> = resize_area(&image, 3, 1).pixels().map(|p| p[0]).collect();
        // Cells of 4/3 pixels: (0*3 + 10)/4 = 2.5 rounds half to even like cvRound
        assert_eq!(row, vec![2, 15, 28]);
    }

    #[test]
    fn perspective_transform_maps_corners() {
        let src = [[10.0, 20.0], [110.0, 25.0], [105.0, 60.0], [12.0, 55.0]];
        let dst = [[0.0, 0.0], [319.0, 0.0], [319.0, 67.0], [0.0, 67.0]];
        let m = perspective_transform(src, dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            let w = m[6] * s[0] as f64 + m[7] * s[1] as f64 + m[8];
            let u = (m[0] * s[0] as f64 + m[1] * s[1] as f64 + m[2]) / w;
            let v = (m[3] * s[0] as f64 + m[4] * s[1] as f64 + m[5]) / w;
            assert!((u - d[0] as f64).abs() < 1e-6 && (v - d[1] as f64).abs() < 1e-6);
        }
    }

    #[test]
    fn identity_warp_keeps_pixels() {
        let image = RgbImage::from_fn(6, 5, |x, y| Rgb([(x * 30) as u8, (y * 40) as u8, 9]));
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(warp_perspective(&image, &identity, 6, 5).unwrap(), image);
    }

    #[test]
    fn lmeds_recovers_similarity_and_ignores_outlier() {
        let truth: Affine = [0.8, -0.3, 12.0, 0.3, 0.8, -5.0];
        let from = [[10.0f32, 20.0], [50.0, 22.0], [30.0, 45.0], [15.0, 70.0], [48.0, 68.0]];
        let mut to: Vec<[f32; 2]> = from
            .iter()
            .map(|p| {
                let (x, y) = (p[0] as f64, p[1] as f64);
                [(truth[0] * x + truth[1] * y + truth[2]) as f32, (truth[3] * x + truth[4] * y + truth[5]) as f32]
            })
            .collect();
        to[4][0] += 40.0;
        let m = estimate_affine_partial_lmeds(&from, &to).unwrap();
        for (a, b) in m.iter().zip(&truth) {
            assert!((a - b).abs() < 1e-4, "{m:?}");
        }
    }

    #[test]
    fn laplacian_variance() {
        let flat = RgbImage::from_pixel(10, 10, Rgb([1, 2, 3]));
        assert_eq!(variance_of_laplacian(&flat), 0.0);
        // A single bright blue pixel inside the measured area: Laplacian -400 there, 100 at its
        // four neighbours, 0 elsewhere over the 4x4 area
        let mut spot = RgbImage::new(10, 10);
        spot.put_pixel(5, 5, Rgb([0, 0, 100]));
        let values = [-400.0f64, 100.0, 100.0, 100.0, 100.0];
        let n = 16.0;
        let mean = values.iter().sum::<f64>() / n;
        let expected = values.iter().map(|v| v * v).sum::<f64>() / n - mean * mean;
        assert_eq!(variance_of_laplacian(&spot), expected);
    }

    #[test]
    fn identity_affine_keeps_pixels() {
        let image = RgbImage::from_fn(6, 5, |x, y| Rgb([(x * 30) as u8, (y * 40) as u8, 9]));
        assert_eq!(warp_affine(&image, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 6, 5), image);
    }

    #[test]
    fn letterbox_pads_to_stride() {
        let image = RgbImage::new(1920, 1080);
        let letterbox = letterbox_yolo_obb(&image, 320);
        assert_eq!((letterbox.width, letterbox.height), (320, 192));
        assert_eq!(letterbox.padding, (0.0, 6.0));
        assert_eq!(letterbox.tensor.len(), 3 * 320 * 192);
    }
}


