import numpy as np
import cv2
import sys
import os

# Các biến cấp module ##########################################################################
MIN_CONTOUR_AREA = 10
RESIZED_IMAGE_WIDTH =20
RESIZED_IMAGE_HEIGHT =30

################################################################################################
def main():
    # Đảm bảo file ảnh tồn tại
    if not os.path.isfile("training_chars.png"):
        print("Lỗi: Không tìm thấy file training_chars.png.")
        sys.exit()

    imgTrainingNumbers = cv2.imread("training_chars.png")  # đọc hình ảnh ký tự dùng để huấn luyện

    # Kiểm tra xem ảnh có đọc thành công không
    if imgTrainingNumbers is None:
        print("Lỗi: Không thể tải ảnh.")
        sys.exit()

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)  # chuyển hình ảnh thành ảnh xám
    imgBlurred = cv2.GaussianBlur(imgGray, (3, 3), 0)  # làm mờ ảnh

    # Lọc ảnh từ ảnh xám sang ảnh đen trắng
    imgThresh = cv2.adaptiveThreshold(
        imgBlurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,# Kích thước vùng lân cận
        2,# Hằng số trừ đi từ trung bình
    )

    cv2.imshow("imgThresh", imgThresh)  # hiển thị ảnh ngưỡng để tham khảo

    imgThreshCopy = imgThresh.copy()  # tạo bản sao của ảnh ngưỡng để tìm contour

    npaContours, hierarchy = cv2.findContours(
        imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.imshow("2copy",imgThreshCopy)

    # Kiểm tra số lượng contour tìm thấy
    print(f"Số lượng contour tìm thấy: {len(npaContours)}")

    # Khai báo mảng numpy trống để lưu ảnh phẳng và danh sách phân loại
    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    intClassifications = []

    # Các ký tự mà ta quan tâm
    intValidChars = [
        ord("0"), ord("1"), ord("2"), ord("3"), ord("4"), ord("5"),
        ord("6"), ord("7"), ord("8"), ord("9"),
        ord("A"), ord("B"), ord("C"), ord("D"), ord("E"), ord("F"),
        ord("G"), ord("H"), ord("I"), ord("J"), ord("K"), ord("L"),
        ord("M"), ord("N"), ord("O"), ord("P"), ord("Q"), ord("R"),
        ord("S"), ord("T"), ord("U"), ord("V"), ord("W"), ord("X"),
        ord("Y"), ord("Z")
    ]

    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            cv2.rectangle(
                imgTrainingNumbers,
                (intX, intY),
                (intX + intW, intY + intH),
                (0, 0, 255),
                2,
            )

            imgROI = imgThresh[intY : intY + intH, intX : intX + intW]  # Cắt ký tự
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("imgROI", imgROI)  # hiển thị ký tự cắt ra
            cv2.imshow("imgROIResized", imgROIResized)  # hiển thị ảnh đã resize
            cv2.imshow("training_numbers.png", imgTrainingNumbers)

            intChar = cv2.waitKey(0)  # chờ người dùng nhấn phím

            if intChar == 27:  # nếu nhấn phím Esc
                sys.exit()
            elif intChar in intValidChars:  # nếu ký tự hợp lệ
                intClassifications.append(intChar)

                # Làm phẳng ảnh thành mảng numpy 1 chiều
                npaFlattenedImage = imgROIResized.reshape(
                    (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)
                )

                # Thêm mảng ảnh phẳng vào mảng tổng
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, axis=0)
            else:
                print("Ký tự không hợp lệ. Vui lòng nhập lại.")
        else:
            print(f"Contour nhỏ hơn kích thước tối thiểu: {cv2.contourArea(npaContour)}")

    # Nếu không có dữ liệu, thông báo lỗi
    if len(intClassifications) == 0 or npaFlattenedImages.size == 0:
        print("Lỗi: Không có dữ liệu nào để ghi vào file.")
        sys.exit()

    fltClassifications = np.array(intClassifications, np.float32)
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

    print("\n\nHoàn thành huấn luyện !!\n")

    # Ghi dữ liệu vào file
    np.savetxt("classifications.txt", npaClassifications)
    np.savetxt("flattened_images.txt", npaFlattenedImages)
    cv2.destroyAllWindows()

    return

###################################################################################################
if __name__ == "__main__":
    main()
