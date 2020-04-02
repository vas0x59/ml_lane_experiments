import cv2
import numpy as np 
from reg_line1 import *
rl = RegLine()
import time 
cap  = cv2.VideoCapture(2)
fourcc = cv2.VideoWriter_fourcc(*'XVID')    
out = cv2.VideoWriter('output.avi',fourcc, 24.0, (360,200))
ret = True
c = 0
ccc = 0
med = 0
while ret:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.resize(frame, (360, 200))
    #cv2.imshow("frame", frame)
    th = rl.thresh(frame)
    # th3 = cv2.adaptiveThreshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #         cv2.THRESH_BINARY_INV,5,2)
    # cv2.imshow("th", th)
    
    e, e2, _, su, img_to, x1, x2, pt= rl.reg_line(frame, show=True)
    print(e, e2, su)
    c+=1
    med += time.time() - pt
    

    # if e == e2 == 0:
    #     print("PASS")
    # else:
    #     ccc += 1
    #     print("GO", ccc)
    #     if True:
    #         cv2.imwrite("data/img" + str(ccc) + ".jpg", img_to)
    #         cv2.imshow("img_to", img_to)
    #         open("data/img" + str(ccc) + ".txt", "w+").write(" ".join(list(map(str, [e, e2, x1, x2]))))
    out.write(img_to)
    if cv2.waitKey(1) == ord('q'):
        break
print(1 / (med / c))
#0.001124315877114573
#full 0.003162892817267709
cv2.destroyAllWindows()
cap.release()
out.release()