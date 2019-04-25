import numpy as np
import cv2
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential

prev_coords = []
maxlen_track = 15

model = Sequential()
model.add(LSTM(64, input_shape=(maxlen_track, 2)))
model.add(Dropout(0.2))
model.add(Dense(2, activation=None))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
filled = 0

points_arr = []
x_ = []
y_ = []
i = 0
draw_flg = False
point_pass = 0
x_past = 0
y_past = 0


def draw_circle(event, x, y, flags, param):
    global x_past, y_past
    global i
    global draw_flg
    global filled
    global point_pass
    if event == cv2.EVENT_MOUSEMOVE and draw_flg:
        if point_pass % 10 == 0:
            points_arr.append([x - x_past, y - y_past])
            cv2.circle(img, (x, y), 5, (255, 0, 0), thickness=1, lineType=8, shift=0)
            x_past = x
            y_past = y
        point_pass += 1
    elif event == cv2.EVENT_LBUTTONDOWN:
        draw_flg = True
    elif event == cv2.EVENT_LBUTTONUP:
        draw_flg = False
        filled = True
        for i in range(0, len(points_arr) - maxlen_track):
            x_.append(points_arr[i: i + maxlen_track])
            y_.append(points_arr[i + maxlen_track])
        global x_in, y_in
        x_in = np.zeros((len(x_), maxlen_track, 2))
        y_in = np.zeros((len(x_), 2))
        for i, coords in enumerate(x_):
            for j, coord in enumerate(coords):
                x_in[i, j] = coord
            y_in[i] = y_[i]

        return


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
while (1):
    cv2.imshow('image', img)
    cv2.rectangle(img,(200,200),(300,300),(0,0,255))
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or filled:
        print("Sad")
        break
cv2.destroyAllWindows()

model.fit(x_in, y_in,
          batch_size=50,
          epochs=100,
          )

x__ = []
x_past = 0
y_past = 0
arr_points = []
point_pass = 0
eval_x = 0
eval_y = 0
img = np.zeros((512, 512, 3), np.uint8)


def res(event, x, y, flags, param):
    global draw_flg, x__, x_past, y_past, point_pass, img, eval_x, eval_y

    if event == cv2.EVENT_MOUSEMOVE and draw_flg:

        if point_pass % 10 == 0:
            img = np.zeros((512, 512, 3), np.uint8)
            x__.append([x - x_past, y - y_past])
            arr_points.append([x, y])
            for point in arr_points:
                cv2.circle(img, (point[0], point[1]), 5, (255, 0, 0), thickness=1, lineType=8, shift=0)
            if len(x__) == maxlen_track:
                eval_x = x
                eval_y = y
                arr_for_pred = x__.copy()
                x_pred = np.zeros((1, maxlen_track, 2))
                for i in range(10):
                    for g, coord__ in enumerate(arr_for_pred):
                        x_pred[0, g] = coord__
                    print (x_pred)
                    pred = list(model.predict(x_pred)[0])

                    arr_for_pred.pop(0)
                    arr_for_pred.append(pred)
                    curr_x = pred[0] + eval_x
                    curr_y = pred[1] + eval_y
                    cv2.circle(img, (int(curr_x), int(curr_y)), 5, (0, 255, 0), thickness=1, lineType=8, shift=0)
                    eval_x = curr_x
                    eval_y = curr_y

                arr_points.pop(0)

                x__.pop(0)

            x_past = x
            y_past = y
        point_pass += 1

    elif event == cv2.EVENT_LBUTTONDOWN:
        draw_flg = True
        point_pass = 0
    elif event == cv2.EVENT_LBUTTONUP:
        draw_flg = False


cv2.namedWindow('image2')
cv2.setMouseCallback('image2', res)
while (1):
    cv2.imshow('image2', img)
    cv2.rectangle(img, (200, 200), (300, 300), (0, 0, 255))
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
