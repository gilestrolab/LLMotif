import cv2
import numpy as np
from optparse import OptionParser
import os
import sys
import time
from datetime import datetime

def detect_circles_in_square(roi, min_radius_factor=4, max_radius_factor=2):
    """Detect circular arena in a square ROI and return its center coordinates and radius."""
    # Convert ROI to grayscale if it's not already
    if len(roi.shape) == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = roi.copy()
    
    # Blur to reduce noise
    gray_roi = cv2.GaussianBlur(gray_roi, (9, 9), 2)
    
    # Calculate min and max radius based on ROI size and provided factors
    min_radius = roi.shape[0] // min_radius_factor  # Smaller factor = larger min radius
    max_radius = roi.shape[0] // max_radius_factor  # Smaller factor = larger max radius
    
    # Use Hough Circle Transform to find circles
    circles = cv2.HoughCircles(
        gray_roi, 
        cv2.HOUGH_GRADIENT, 
        dp=1,
        minDist=roi.shape[0] // 2,  # Only one circle expected per ROI
        param1=100,                 # Higher threshold for edge detection
        param2=30,                  # Lower threshold means more false circles
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # If circles are found, return the first one
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Return center x, y and radius
        return circles[0, 0]  # First circle [x, y, radius]
    
    # If no circles found, return center of ROI and estimated radius
    h, w = gray_roi.shape[:2]
    return np.array([w//2, h//2, min(w, h)//3], dtype=np.uint16)

def center_square_on_circle(img, x, y, w, h, min_radius_factor=4, max_radius_factor=2, size_multiplier=3.75):
    """Create a square ROI centered on the circle within the given bounding box."""
    # First, find circle in the initial ROI
    roi = img[y:y+h, x:x+w].copy()
    circle = detect_circles_in_square(roi, min_radius_factor, max_radius_factor)
    
    # Get circle center coordinates relative to original image
    center_x = x + circle[0]
    center_y = y + circle[1]
    radius = circle[2]
    
    # Create a square ROI centered on the circle
    square_size = int(size_multiplier * radius)
    
    # Calculate ROI coordinates
    x1 = max(0, center_x - square_size // 2)
    y1 = max(0, center_y - square_size // 2)
    x2 = min(img.shape[1], center_x + square_size // 2)
    y2 = min(img.shape[0], center_y + square_size // 2)
    
    # Extract the square ROI
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def find_and_segment_arenas(frame, threshold_method='otsu', blur_size=5, 
                          circle_min_radius_factor=4, circle_max_radius_factor=2, 
                          size_multiplier=3.75):
    """Find and segment arenas in a video frame."""
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Apply thresholding based on method
    if threshold_method == 'otsu':
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif threshold_method == 'adaptive':
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    else:  # Manual threshold
        try:
            threshold_value = int(threshold_method)
            _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        except ValueError:
            print(f"Invalid threshold method: {threshold_method}. Using Otsu.")
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find square contours
    square_contours = []
    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(approx)
        
        # Check if it's approximately a quadrilateral and above minimum size
        if len(approx) == 4 and area > 1000:
            square_contours.append(approx)
    
    # Sort squares by position
    def contour_position(approx):
        x, y, w, h = cv2.boundingRect(approx)
        return (y//100, x)  # Group by rows, then sort by x
    
    square_contours.sort(key=contour_position)
    
    # Process each square to find circles and center ROIs
    cropped_images = []
    circle_roi_coords = []
    
    for sq in square_contours:
        x, y, w, h = cv2.boundingRect(sq)
        
        # Get centered ROI
        centered_roi, (x1, y1, x2, y2) = center_square_on_circle(
            frame, x, y, w, h, 
            min_radius_factor=circle_min_radius_factor, 
            max_radius_factor=circle_max_radius_factor,
            size_multiplier=size_multiplier
        )
        
        cropped_images.append(centered_roi)
        circle_roi_coords.append((x1, y1, x2, y2))
    
    return {
        'cropped_images': cropped_images,
        'circle_roi_coords': circle_roi_coords,
        'contours': square_contours
    }

def process_video(video_path, output_dir="output", time_interval=1.0, resolution=None,
                  threshold_method='otsu', blur_size=5,
                  circle_min_radius_factor=4, circle_max_radius_factor=2,
                  size_multiplier=3.75):
    """
    Process a video file, extract frames at specified time intervals,
    segment each frame into ROIs, and save the results.
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save output frames and ROIs
        time_interval: Time interval in seconds between extracted frames (default: 1 second)
        resolution: Optional (width, height) to resize frames
        threshold_method: Method for thresholding ('otsu', 'adaptive', or number)
        blur_size: Gaussian blur kernel size
        circle_min_radius_factor: Factor for minimum circle radius
        circle_max_radius_factor: Factor for maximum circle radius
        size_multiplier: Size multiplier for ROIs
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / original_fps
    print(f"Video info: {total_frames} frames, {original_fps} fps, duration: {video_duration:.2f} seconds")
    
    # Extract video filename (without extension)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output directories with new structure
    video_output_dir = os.path.join(output_dir, video_basename)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    
    # Create subdirectories for ROIs and snapshots
    rois_dir = os.path.join(video_output_dir, "roi")
    snapshots_dir = os.path.join(video_output_dir, "snapshots")
    
    if not os.path.exists(rois_dir):
        os.makedirs(rois_dir)
    if not os.path.exists(snapshots_dir):
        os.makedirs(snapshots_dir)
    
    # Calculate frame interval based on time interval
    frame_interval = int(original_fps * time_interval)
    
    # Ensure we capture at least one frame
    frame_interval = max(1, frame_interval)
    
    # Initialize frame counter
    frame_count = 0
    processed_count = 0
    
    if time_interval == 1.0:
        interval_description = "1 second"
    else:
        interval_description = f"{time_interval} seconds"
    
    print(f"Processing video: extracting 1 frame every {interval_description}")
    
    # Process the first frame to identify arenas
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return
    
    # Resize if resolution specified
    if resolution:
        frame = cv2.resize(frame, resolution)
    
    # Find arenas in the first frame
    first_frame_results = find_and_segment_arenas(
        frame, threshold_method, blur_size,
        circle_min_radius_factor, circle_max_radius_factor,
        size_multiplier
    )
    
    # We'll use these arena coordinates for all subsequent frames
    arena_coords = first_frame_results['circle_roi_coords']
    
    # Check if we found enough arenas (should be 6)
    if len(arena_coords) != 6:
        print(f"Warning: Found {len(arena_coords)} arenas instead of the expected 6")
    
    # Reset to beginning of video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Start processing
    start_time = time.time()
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Increment frame counter
        frame_count += 1
        
        # Skip frames to maintain desired interval
        if (frame_count - 1) % frame_interval != 0:
            continue
        
        # Resize if resolution specified
        if resolution:
            frame = cv2.resize(frame, resolution)
        
        # Increment processed counter
        processed_count += 1
        
        # Format timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Save the full frame to snapshots directory
        full_frame_filename = f"{video_basename}_frame_{processed_count:04d}_{timestamp}.jpg"
        full_frame_path = os.path.join(snapshots_dir, full_frame_filename)
        cv2.imwrite(full_frame_path, frame)
        
        # Extract and save each ROI using the coordinates from the first frame
        for i, (x1, y1, x2, y2) in enumerate(arena_coords):
            # Extract ROI
            roi = frame[y1:y2, x1:x2].copy()
            
            # Save ROI to roi directory
            roi_filename = f"{video_basename}_frame_{processed_count:04d}_arena_{i+1}_{timestamp}.jpg"
            roi_path = os.path.join(rois_dir, roi_filename)
            cv2.imwrite(roi_path, roi)
        
        # Print progress every 10 frames
        if processed_count % 10 == 0:
            elapsed_time = time.time() - start_time
            percent_done = (frame_count / total_frames) * 100
            print(f"Processed {processed_count} frames ({percent_done:.1f}% of video)")
    
    # Clean up
    cap.release()
    print(f"Completed processing {processed_count} frames from {video_path}")
    print(f"Output saved to {video_output_dir}")
    print(f"- Snapshots: {snapshots_dir}")
    print(f"- ROIs: {rois_dir}")
    
def main():
    parser = OptionParser()
    parser.add_option(
        "-f", "--file",
        dest="filename",
        default=None,
        help="Path to the input video file",
        metavar="FILE"
    )
    parser.add_option(
        "-o", "--output-dir",
        dest="output_dir",
        default="output",
        help="Directory for output images (optional)",
        metavar="DIR"
    )
    parser.add_option(
        "--interval",
        dest="time_interval",
        type="float",
        default=1.0,
        help="Time interval in seconds between frames (e.g., 60 for one frame per minute, default: 1)"
    )
    parser.add_option(
        "--width",
        dest="width",
        type="int",
        default=None,
        help="Resize frames to this width (optional)"
    )
    parser.add_option(
        "--height",
        dest="height",
        type="int",
        default=None,
        help="Resize frames to this height (optional)"
    )
    parser.add_option(
        "-t", "--threshold",
        dest="threshold_method",
        default="otsu",
        help="Threshold method: 'otsu', 'adaptive', or a number for manual threshold"
    )
    parser.add_option(
        "-b", "--blur",
        dest="blur_size",
        type="int",
        default=5,
        help="Gaussian blur kernel size (odd number)"
    )
    parser.add_option(
        "--min-radius-factor",
        dest="min_radius_factor",
        type="int",
        default=4,
        help="Factor to divide ROI size by to get minimum circle radius (smaller = larger circles)"
    )
    parser.add_option(
        "--max-radius-factor",
        dest="max_radius_factor",
        type="int",
        default=2,
        help="Factor to divide ROI size by to get maximum circle radius (smaller = larger circles)"
    )
    parser.add_option(
        "--size-multiplier",
        dest="size_multiplier",
        type="float",
        default=3.75,
        help="Multiplier for the square size relative to circle radius"
    )
    (options, args) = parser.parse_args()
    
    # Allow for using first argument as filename without a switch
    if not options.filename and len(args) > 0:
        options.filename = args[0]
    
    # If we still don't have a filename, show error
    if not options.filename:
        print("You must specify a video file with -f, --file, or as the first argument.")
        parser.print_help()
        return
    
    # Set resolution if both width and height are specified
    resolution = None
    if options.width is not None and options.height is not None:
        resolution = (options.width, options.height)
    
    # Process the video
    process_video(
        options.filename,
        output_dir=options.output_dir,
        time_interval=options.time_interval,
        resolution=resolution,
        threshold_method=options.threshold_method,
        blur_size=options.blur_size,
        circle_min_radius_factor=options.min_radius_factor,
        circle_max_radius_factor=options.max_radius_factor,
        size_multiplier=options.size_multiplier
    )

if __name__ == "__main__":
    main()