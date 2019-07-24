import cv2


def mouse_callback(event, x, y, flags, params):
    """
    This function will be called whenever the mouse is clicked.
    :param event: click event type
    :param x: click point's x pixel value.
    :param y: click point's y pixel value.
    :param flags:
    :param params:
    :return:
    """
    if event == cv2.EVENT_FLAG_LBUTTON:
        print("mouse left click")
    elif event == cv2.EVENT_FLAG_RBUTTON:
        print("mouse right click")
        # # store the coordinates of the right-click event
        right_clicks.append([x, y])
        print(right_clicks)
        # plot a circle. (img, (x, y), radius, (b, g, r), -1)
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow('image', img)
        cv2.imwrite('./logs/pcb_a_c_click.png', img)
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        print("mouse left double click")
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        print("mouse right double click")
    else:
        pass


def main():
    # # print all events defined in opencv.
    # events = [i for i in dir(cv2) if 'EVENT' in i]
    # print(events)
    global img, right_clicks
    right_clicks = list()
    img = cv2.imread('./logs/pcb_a_c.png')

    window_width = int(img.shape[1])
    window_height = int(img.shape[0])
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)

    # set mouse callback function for window
    cv2.setMouseCallback('image', mouse_callback)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
