lane_x_start, lane_x_end = 200, 450
lane_y_start, lane_y_end = 200, 400

def is_in_lane(x, y, w, h):
    center_x, center_y = x + w // 2, y + h // 2
    return lane_x_start < center_x < lane_x_end and lane_y_start < center_y < lane_y_end
