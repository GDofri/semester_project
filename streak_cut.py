import numpy as np
import cv2 as cv
import gradient_image as gi


# def cut_streak(streakName: str, extension: int, id: int) -> np.ndarray:
#     streaks_csv = playground.read_streaks_csv()
#     streak = streaks_csv[(streaks_csv["file_name"] == streakName) &
#                          (streaks_csv["extension"] == extension) &
#                          (streaks_csv["ID"] == id)].iloc[0]
#
#     streak_x_start = streak["x_start[px]"].astype(np.int8)
#     streak_y_start = streak["y_start[px]"].astype(np.int8)
#     streak_x_end = streak["x_end[px]"].astype(np.int8)
#     streak_y_end = streak["y_end[px]"].astype(np.int8)
#     img = fitsio.read(playground.get_fits_path(streakName), ext=extension)
#     # img = img[streak_y_start:streak_y_end, streak_x_start:streak_x_end]
#     # center = (streak_x_start + streak_x_end) // 2, (streak_y_start + streak_y_end) // 2
#     # angle = np.arctan2(streak_y_end - streak_y_start, streak_x_end - streak_x_start)
#     # M = cv.getRotationMatrix2D(center, angle, 1)
#     # img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
#     return cut_around_line(img, (streak_x_start, streak_y_start), (streak_x_end, streak_y_end), 16)
#
#     # return thin_streak


def cut_around_line(img: np.ndarray, point1: tuple, point2: tuple, thickness: int, trim : bool = False ) -> np.ndarray:

    if trim:
        # Throw not implemented error
        raise NotImplementedError("Trimming not implemented yet")
    streak_x_start, streak_y_start = point1
    streak_x_end, streak_y_end = point2
    # streak_x_end += 1
    # streak_y_end += 1
    center = (streak_x_start + streak_x_end) // 2, (streak_y_start + streak_y_end) // 2
    radians = np.arctan2(streak_y_end - streak_y_start, streak_x_end - streak_x_start)
    degrees = np.rad2deg(radians)
    M = cv.getRotationMatrix2D(center, degrees, 1)
    # trim_size = round((thickness / 2) / np.tan(min(radians, np.pi / 2 - radians)))
    # trim_size = 40
    # print(trim_size)
    # print(np.tan(min(radians, np.pi / 2 - radians)))
    # diagonal -= trim_size * 2 - 10
    new_start_point = np.dot(M, np.array([streak_x_start, streak_y_start, 1]))
    new_end_point = np.dot(M, np.array([streak_x_end, streak_y_end, 1]))
    new_width = np.abs(new_end_point[0]+1 - new_start_point[0]).round().astype(int)
    new_center = (new_width // 2, thickness // 2)
    diff = np.array(new_center) - np.array(center)
    M[0, 2] += diff[0]
    M[1, 2] += diff[1]
    img = cv.warpAffine(img, M, (new_width, thickness))

    return img


def main():
    streaks_csv = playground.read_streaks_csv()

    width = 200
    height = 100
    point1 = (0, 0)
    point2 = (199, 99)
    img = gi.create_gradient_image_with_line(height, width, point1, point2)
    cv.imshow("Original", img)
    rotated = cut_around_line(img, point1, point2, 16)
    cv.imshow("Rotated", rotated)
    cv.waitKey(0)
    streaks_csv = playground.read_streaks_csv()

    # for(_, streak) in list(streaks_csv.iterrows())[:10]:
    #     streakName = streak["file_name"]
    #     extension = streak["extension"]
    #     id = streak["ID"]
    #     img = cut_streak(streakName, extension, id)
    #     cv.imshow("Streak", img)
    #     cv.waitKey(0)

if __name__ == "__main__":
    main()
