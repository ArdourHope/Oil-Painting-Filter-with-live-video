import cv2
import keyboard
import numpy as np

cam = cv2.VideoCapture(0)

cv2.namedWindow("Camera")

img_counter = 0

def apply_invert(frame):
    return cv2.bitwise_not(frame)

def apply_sharp(frame):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(frame, -1, kernel)

def grayscale(frame):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def oilpaint(frame):
    color_image = cv2.imread(frame, cv2.IMREAD_COLOR)

    height = color_image.shape[0]
    width  = color_image.shape[1]
    channel = color_image.shape[2]

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    oil_image = np.zeros((height, width, channel))

    R = 4 
    stroke_mask = []
    for y in range(-R, R):
        for x in range(-R, R):
            if y*y + x*x < R*R:
                stroke_mask.append( (y,x) )

    for y in range(height):
        for x in range(width):
            progress = np.round(100*(y*width+x)/(width*height), 2)
            print( "Progress: ", str(progress)+"%", end='\r' )
            local_histogram = np.zeros(256)
            local_channel_count = np.zeros((channel, 256))
            for dy,dx in stroke_mask:
                yy = y+dy
                xx = x+dx
                if yy < 0  or yy >= height or xx <= 0  or xx >= width:
                    continue
                intensity = gray_image[yy, xx]
                local_histogram[intensity] += 1
                for c in range(channel):
                    local_channel_count[c, intensity] += color_image[yy, xx, c]

            max_intensity = np.argmax(local_histogram)
            max_intensity_count = local_histogram[max_intensity]
            for c in range(channel):
                oil_image[y,x,c] = local_channel_count[c, max_intensity] / max_intensity_count

    print()
    oil_image = oil_image.astype('int')
    cv2.imwrite("result.jpg", oil_image)

def main():
    img_counter = 0
    while True:
        
        if keyboard.is_pressed("2"):
            print ("key pressed")
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    break
                invert = apply_invert(frame)
                cv2.imshow("Invert", invert)

                k = cv2.waitKey(1)
                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    cv2.destroyAllWindows()
                    break

                elif k%256 == 32:
                    # SPACE pressed
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    oilpaint(img_name)
                    img_counter += 1

                
        elif keyboard.is_pressed("1"):
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    break
                cv2.imshow("Normal", frame)

                k = cv2.waitKey(1)
                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    cv2.destroyAllWindows()
                    break
                elif k%256 == 32:
                    # SPACE pressed
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    oilpaint(img_name)

                    img_counter += 1

        elif keyboard.is_pressed("3"):
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    break
                sharp = apply_sharp(frame)
                cv2.imshow("Sharpening", sharp)

                k = cv2.waitKey(1)
                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    cv2.destroyAllWindows()
                    break
                elif k%256 == 32:
                    # SPACE pressed
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    oilpaint(img_name)
                    img_counter += 1

        elif keyboard.is_pressed("4"):
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    break
                gray = grayscale(frame)
                cv2.imshow("gray", gray)

                k = cv2.waitKey(1)
                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    cv2.destroyAllWindows()
                    break
                elif k%256 == 32:
                    # SPACE pressed
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    oilpaint(img_name)
                    img_counter += 1

        elif keyboard.is_pressed("q"):
            print("Escape hit, closing...")
            break
                
        # elif keyboard.is_pressed("2")
        #     cam.release()
        #     ret, frame = cam.read()
        #     invert = apply_invert(frame)


    cam.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()