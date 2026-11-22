/** Shared computer-vision types. All image coordinates are pixels with y pointing down. */

export interface Vec2 {
  x: number;
  y: number;
}

/** Axis-aligned box, top-left origin. */
export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Detection {
  bbox: BBox;
  /** Final confidence (objectness × class probability for YOLOX). */
  score: number;
  classId: number;
}

export type Polygon = Vec2[];

/** A crop of the source frame that was resized into the model input. */
export interface CropRegion {
  x: number;
  y: number;
  w: number;
  h: number;
}
