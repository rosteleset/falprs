//! Box and polygon geometry of the recognition pipeline, matching OpenCV's `Rect2f` and
//! `intersectConvexConvex` as the C++ service uses them.

/// `[x1, y1, x2, y2]`.
pub type BBox = [f32; 4];

/// A box normalised like `cv::Rect2f(Point2f, Point2f)`: `[x, y, width, height]`.
fn rect(b: &BBox) -> [f32; 4] {
    [b[0].min(b[2]), b[1].min(b[3]), (b[2] - b[0]).abs(), (b[3] - b[1]).abs()]
}

pub fn area(b: &BBox) -> f32 {
    let r = rect(b);
    r[2] * r[3]
}

/// Intersection over union; NaN for two empty boxes, like the C++ service.
pub fn iou(a: &BBox, b: &BBox) -> f32 {
    let (ra, rb) = (rect(a), rect(b));
    let x1 = ra[0].max(rb[0]);
    let y1 = ra[1].max(rb[1]);
    let w = (ra[0] + ra[2]).min(rb[0] + rb[2]) - x1;
    let h = (ra[1] + ra[3]).min(rb[1] + rb[3]) - y1;
    let intersection = if w <= 0.0 || h <= 0.0 { 0.0 } else { w * h };
    intersection / (ra[2] * ra[3] + rb[2] * rb[3] - intersection)
}

/// Whether two boxes overlap with a positive area.
pub fn has_intersection(a: &BBox, b: &BBox) -> bool {
    a[0].max(b[0]) < a[2].min(b[2]) && a[1].max(b[1]) < a[3].min(b[3])
}

pub fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])).sqrt()
}

fn signed_area(polygon: &[[f64; 2]]) -> f64 {
    let n = polygon.len();
    (0..n)
        .map(|i| {
            let (p, q) = (polygon[i], polygon[(i + 1) % n]);
            p[0] * q[1] - q[0] * p[1]
        })
        .sum::<f64>()
        / 2.0
}

/// The intersection of two convex polygons (Sutherland-Hodgman clipping).
fn intersect_convex(subject: &[[f32; 2]], clip: &[[f32; 2]]) -> Vec<[f64; 2]> {
    let to_f64 = |p: &[f32; 2]| [p[0] as f64, p[1] as f64];
    let mut clip: Vec<[f64; 2]> = clip.iter().map(to_f64).collect();
    if signed_area(&clip) < 0.0 {
        clip.reverse();
    }
    let mut output: Vec<[f64; 2]> = subject.iter().map(to_f64).collect();
    for i in 0..clip.len() {
        if output.is_empty() {
            break;
        }
        let (a, b) = (clip[i], clip[(i + 1) % clip.len()]);
        let inside = |p: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0;
        let crossing = |p: [f64; 2], q: [f64; 2]| {
            let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
            let denominator = (b[0] - a[0]) * dy - (b[1] - a[1]) * dx;
            let t = ((b[1] - a[1]) * (p[0] - a[0]) - (b[0] - a[0]) * (p[1] - a[1])) / denominator;
            [p[0] + t * dx, p[1] + t * dy]
        };
        let input = std::mem::take(&mut output);
        for j in 0..input.len() {
            let (current, previous) = (input[j], input[(j + input.len() - 1) % input.len()]);
            match (inside(current), inside(previous)) {
                (true, true) => output.push(current),
                (true, false) => {
                    output.push(crossing(previous, current));
                    output.push(current);
                }
                (false, true) => output.push(crossing(previous, current)),
                (false, false) => {}
            }
        }
    }
    output
}

/// Area of the intersection of two convex polygons.
pub fn intersection_area(a: &[[f32; 2]], b: &[[f32; 2]]) -> f32 {
    let polygon = intersect_convex(a, b);
    if polygon.len() < 3 {
        return 0.0;
    }
    signed_area(&polygon).abs() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxes() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 0.0, 15.0, 10.0];
        assert!((iou(&a, &b) - 50.0 / 150.0).abs() < 1e-6);
        assert!(has_intersection(&a, &b));
        assert!(!has_intersection(&a, &[10.0, 0.0, 20.0, 10.0]));
        assert_eq!(iou(&a, &[20.0, 20.0, 30.0, 30.0]), 0.0);
        assert_eq!(area(&[10.0, 10.0, 0.0, 0.0]), 100.0);
    }

    #[test]
    fn polygons() {
        let square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let shifted = [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]];
        let reversed = [[0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]];
        let inner = [[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]];
        assert_eq!(intersection_area(&square, &square), 100.0);
        assert_eq!(intersection_area(&square, &shifted), 25.0);
        assert_eq!(intersection_area(&shifted, &reversed), 25.0);
        assert_eq!(intersection_area(&inner, &square), 4.0);
        assert_eq!(intersection_area(&square, &[[20.0, 20.0], [30.0, 20.0], [30.0, 30.0]]), 0.0);
    }
}
