
import cv2
import numpy as np
import tensorflow as tf
import catboost
import time
keras = tf.keras
src = np.float32([[0, 200],
                  [360, 200],
                  [300, 120],
                  [60, 120]])

# self.src = np.float32([[0, 299],
#            [399, 299],
#            [320, 200],
#            [80, 200]])
img_size = [200, 360]
src_draw=np.array(src,dtype=np.int32)

dst = np.float32([[0, img_size[0]],
                [img_size[1], img_size[0]],
                [img_size[1], 0],
                [0, 0]])

encoder  = keras.models.load_model('models/keras_autoencoder/v1_1585822666/v1_1585822666_encoder.keras')
decoder  = keras.models.load_model('models/keras_autoencoder/v1_1585822666/v1_1585822666_decoder.keras')
cat_x1 = catboost.CatBoostRegressor()
cat_x2 = catboost.CatBoostRegressor()
cat_x1.load_model("models/catboost_decoder/v1_1585822784/v1_1585822784x1.catboost")
cat_x2.load_model("models/catboost_decoder/v1_1585822784/v1_1585822784x2.catboost")
M = cv2.getPerspectiveTransform(src, dst)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_7.avi',fourcc, 24.0, (360,200))
# ssss = 0
cap  = cv2.VideoCapture(2)
# for i in range(0, 1665):
c = 0
med = 0
while cv2.waitKey(1) != ord("q"):
    ret, img_to_test_out = cap.read()
    img_to_test_out = cv2.resize(img_to_test_out, (360, 200))
    img_to_test_out = cv2.warpPerspective(img_to_test_out, M, (img_size[1],img_size[0]), flags=cv2.INTER_LINEAR)

    if ret == False:
        break
    # ssss+=1
    # print(ssss)
    # iiid = i

    #img_to_test_out = data_v1[iiid][1].copy()
    
    enc_out = encoder.predict(np.array([np.expand_dims(cv2.cvtColor(cv2.resize(img_to_test_out, (64, 64)), cv2.COLOR_BGR2GRAY), -1)/255]))
    # enc_out = encoder.predict(np.array([x_data_r8050[0]]))[0]
    pt = time.time()
    x1l = cat_x1.predict(enc_out)
    x2l = cat_x2.predict(enc_out)
    c+=1
    med += time.time() - pt
    # img_to_test_out
#     img_to_test_out = img_to_test.copy()
    cv2.circle(img_to_test_out,(int(img_to_test_out.shape[1]/2 - (img_to_test_out.shape[1]) * x1l[0]),int(img_to_test_out.shape[0])),4,(0,255,255),2)
    cv2.circle(img_to_test_out,(int(img_to_test_out.shape[1]/2 - (img_to_test_out.shape[1]) * x2l[0]),int(img_to_test_out.shape[0] // 8 *3)),4,(0,80,255),2)
#     print(img_to_test_out.shape)
#     if not i % 12 == 0:

    cv2.imshow("IMG", img_to_test_out)
    cv2.imshow("AAAAA_PREDICTED", cv2.resize(decoder.predict(np.array(enc_out))[0], (360, 200)))
    out.write(img_to_test_out)
print(1 / (med / c))
out.release()
cap.release()