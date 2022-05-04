#Optical-Mark-Recognition with OpenCV
#Original_ref: https://github.com/murtazahassan/Optical-Mark-Recognition-OPENCV
#from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import utlis

########################################################################
pathImage = "Data/Photo/IMG_5.jpg"
heightImg = 900
widthImg  = 700
gridH = 21
gridW = 5
rlQuestions = gridH - 1
rlChoices = gridW - 1
Data = np.genfromtxt("Data/Answer_Key/AK1.txt", dtype=int,
                     encoding=None, delimiter=",")
ans1 = Data[0]
ans2 = Data[1]

#FIND TOTAL ANSWER KEY (IGNORE ZERO)
Tans1 = np.delete(Data[0], np.where(Data[0] < 1)[0], axis=0)
Tans2 = np.delete(Data[1], np.where(Data[1] < 1)[0], axis=0)
ttlQuestions = len(Tans1)+len(Tans2)
########################################################################

img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))
imgFinal = img.copy()
imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) 
imgCanny = cv2.Canny(imgBlur,10,70) 

## FIND ALL COUNTOURS
imgContours = img.copy()
imgBigContour = img.copy()
imgBigContourB = img.copy() 
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) 
rectCon = utlis.rectContour(contours)
biggestPoints= utlis.getCornerPoints(rectCon[0]) # GET CORNER POINTS OF THE BIGGEST RECTANGLE
biggestPointsB= utlis.getCornerPoints(rectCon[1]) # GET CORNER POINTS OF THE 2nd BIGGEST RECTANGLE
gradePoints = utlis.getCornerPoints(rectCon[2]) # GET CORNER POINTS OF THE 3RD BIGGEST RECTANGLE

#cv2_imshow(imgContours)

if biggestPoints.size != 0 and gradePoints.size != 0:

    # BIGGEST RECTANGLE WARPING
    biggestPoints=utlis.reorder(biggestPoints) # REORDER FOR WARPING
    cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggestPoints) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GET TRANSFORMATION MATRIX
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # APPLY WARP PERSPECTIVE

    # 2nd BIGGEST RECTANGLE WARPING
    biggestPointsB=utlis.reorder(biggestPointsB) # REORDER FOR WARPING
    cv2.drawContours(imgBigContourB, biggestPointsB, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
    pts1B = np.float32(biggestPointsB) # PREPARE POINTS FOR WARP
    pts2B = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrixB = cv2.getPerspectiveTransform(pts1B, pts2B) # GET TRANSFORMATION MATRIX
    imgWarpColoredB = cv2.warpPerspective(img, matrixB, (widthImg, heightImg)) # APPLY WARP PERSPECTIVE

    # GRADING RECTANGLE WARPING
    cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20) # DRAW THE BIGGEST CONTOUR
    gradePoints = utlis.reorder(gradePoints) # REORDER FOR WARPING
    ptsG1 = np.float32(gradePoints)  # PREPARE POINTS FOR WARP
    ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # PREPARE POINTS FOR WARP
    matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)# GET TRANSFORMATION MATRIX
    imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150)) # APPLY WARP PERSPECTIVE

    # APPLY THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
    imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE

    # APPLY THRESHOLD_B
    imgWarpGrayB = cv2.cvtColor(imgWarpColoredB,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
    imgThreshB = cv2.threshold(imgWarpGrayB, 170, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE

    boxes = utlis.splitBoxesFree(imgThresh,gridW,gridH) # GET INDIVIDUAL BOXES
    boxesB = utlis.splitBoxesFree(imgThreshB,gridW,gridH) # GET INDIVIDUAL BOXES

LstBigRect = [(cv2.countNonZero(boxes[5]), cv2.countNonZero(boxesB[5]))]


# 1st BIGGEST RECT.
countR=0
countC=0
myPixelVal = np.zeros((gridH, gridW)) # TO STORE THE NON ZERO VALUES OF EACH BOX
for image in boxes:
    totalPixels = cv2.countNonZero(image)
    myPixelVal[countR][countC]= totalPixels
    countC += 1
    if (countC==gridW):countC=0;countR +=1

# DEFINE THE ANSWER KEY 1(1-20) OR 2(21-40)
ans = []
if cv2.countNonZero(boxes[5]) == np.amin(LstBigRect):
  ans = ans1
else:ans = ans2

# FIND THE ANSWER INDEX
arr = np.asarray(ans)
nonZeroAns = np.delete(ans, np.where(ans < 1)[0], axis=0)
LnonZeroAns = len(nonZeroAns)
myIndex=[]
for x in range (1,LnonZeroAns+1):
    arr = myPixelVal[x]
    myIndexVal = np.where(arr == np.amax(arr[1:]))
    myIndex.append(myIndexVal[0][0])

# GRADE THE ANSWER INDEX
grading=[]
for x in range(0,LnonZeroAns):
    if ans[x] == myIndex[x]:
      grading.append(1)
    else:grading.append(0)

utlis.showAnswers(imgWarpColored,myIndex,grading,ans,LnonZeroAns) # DRAW DETECTED ANSWERS
utlis.drawGrid(imgWarpColored,gridH,gridW) # DRAW GRID
imgRawDrawings = np.zeros_like(imgWarpColored) # NEW BLANK IMAGE WITH WARP IMAGE SIZE
utlis.showAnswers(imgRawDrawings, myIndex, grading, ans, LnonZeroAns) # DRAW ON NEW IMAGE
invMatrix = cv2.getPerspectiveTransform(pts2, pts1) # INVERSE TRANSFORMATION MATRIX
imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg)) # INV IMAGE WARP
#cv2.imshow(imgWarpColored)


# 2nd BIGGEST RECT.
countR=0
countC=0
myPixelValB = np.zeros((gridH, gridW)) # TO STORE THE NON ZERO VALUES OF EACH BOX
for image in boxesB:
    totalPixels = cv2.countNonZero(image)
    myPixelValB[countR][countC]= totalPixels
    countC += 1
    if (countC==gridW):countC=0;countR +=1

ans = []
if cv2.countNonZero(boxesB[5]) == np.amin(LstBigRect):
  ans = ans1
else:ans = ans2

arr = np.asarray(ans)
nonZeroAns = np.delete(ans, np.where(ans < 1)[0], axis=0)
LnonZeroAns = len(nonZeroAns)
myIndexB=[]
for x in range (1,LnonZeroAns+1):
    arr = myPixelValB[x]
    myIndexValB = np.where(arr == np.amax(arr[1:]))
    myIndexB.append(myIndexValB[0][0])

gradingB=[]
for x in range(0,LnonZeroAns):
    if ans[x] == myIndexB[x]:
      gradingB.append(1)
    else:gradingB.append(0)

utlis.showAnswers(imgWarpColoredB,myIndexB,gradingB,ans,LnonZeroAns) # DRAW DETECTED ANSWERS
utlis.drawGrid(imgWarpColoredB,gridH,gridW) # DRAW GRID
imgRawDrawingsB = np.zeros_like(imgWarpColoredB) # NEW BLANK IMAGE WITH WARP IMAGE SIZE
utlis.showAnswers(imgRawDrawingsB, myIndexB, gradingB, ans, LnonZeroAns) # DRAW ON NEW IMAGE
invMatrixB = cv2.getPerspectiveTransform(pts2B, pts1B) # INVERSE TRANSFORMATION MATRIX
imgInvWarpB = cv2.warpPerspective(imgRawDrawingsB, invMatrixB, (widthImg, heightImg)) # INV IMAGE WARP
#cv2.imshow(imgWarpColoredB)


Fscore = ((sum(grading)+sum(gradingB))/ttlQuestions)*100 # FINAL GRADE
#print("Final Score",Fscore)


# DISPLAY GRADE
imgRawGrade = np.zeros_like(imgGradeDisplay,np.uint8) # NEW BLANK IMAGE WITH GRADE AREA SIZE
cv2.putText(imgRawGrade,str(int(Fscore)),(70,100)
            ,cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3) # ADD THE GRADE TO NEW IMAGE
invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1) # INVERSE TRANSFORMATION MATRIX
imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg)) # INV IMAGE WARP


# SHOW ANSWERS AND GRADE ON FINAL IMAGE
imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarpB, 1,0)
imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)


# IMAGE ARRAY FOR DISPLAY
cv2.imshow("Final", imgFinal)
#cv2.imshow("Final", imgContours)
#cv2.imshow("Final", imgWarpColoredB)
#cv2.imshow("Final", imgCanny)
print(myIndex)


while True:
  k = cv2.waitKey(1)
  if k%256 == 27:
    # ESC pressed
    print("Escape hit, closing...")
    break
