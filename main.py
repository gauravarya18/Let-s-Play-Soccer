from PIL import Image
from IPython.display import display
import cv2
import random
import numpy as np

def markingPlayers(starting,starting1,ending,ending1,halfStarting,halfStarting1,halfEnding,halfEnding1):
    
    image = cv2.imread("C:/Users/Gaurav Arya/Desktop/Assignment 1/AI2_Assignment1_T3_2021.PNG")

    center_coordinates = (120, 50)
    radius = 20
    color = (255, 0, 0)
    thickness = -1

    B1 = (random.randint(starting+20,ending-20),random.randint(starting1+20,ending1-20))
    R1 = (random.randint(starting+20,ending-20),random.randint(starting1+20,ending1-20))
    while(True):
        point1 = np.array(R1)
        point2 = np.array(B1)
        dist1 = np.linalg.norm(point1 - point2)
        if(dist1>2*radius):
            break
        R1 = (random.randint(starting+20,ending-20),random.randint(starting1+20,ending1-20))

    image = cv2.circle(image,B1, radius, (255, 0, 0), thickness)
    image = cv2.circle(image,R1 , radius, (0,0,255), thickness)


    B2 = (random.randint(halfStarting+20,halfEnding-20),random.randint(halfStarting1+20,halfEnding1-20))
    R2 = (random.randint(halfStarting+20,halfEnding-20),random.randint(halfStarting1+20,halfEnding1-20))
    while(True):
        point1 = np.array(R2)
        point2 = np.array(B2)
        dist1 = np.linalg.norm(point1 - point2)
        if(dist1>2*radius):
            break
        R2 = (random.randint(halfStarting+20,halfEnding-20),random.randint(halfStarting1+20,halfEnding1-20))
    image = cv2.circle(image, B2 , radius, (255, 0, 0), thickness)
    image = cv2.circle(image, R2, radius, (0,0,255), thickness)

    B3 = (random.randint(halfStarting+20,halfEnding-20),random.randint(halfStarting1+20,halfEnding1-20))
    while(True):
        point1 = np.array(R2)
        point2 = np.array(B2)
        point3 = np.array(B3)
        dist1 = np.linalg.norm(point1 - point3)
        dist2 = np.linalg.norm(point2 - point3)

        if(dist1>2*radius and dist2>2*radius):
            break
        B3 = (random.randint(halfStarting+20,halfEnding-20),random.randint(halfStarting1+20,halfEnding1-20))
    
    R3 = (random.randint(halfStarting+20,halfEnding-20),random.randint(halfStarting1+20,halfEnding1-20))
    while(True):
        point1 = np.array(R2)
        point2 = np.array(B3)
        point3 = np.array(B2)
        point4 = np.array(R3)
        dist1 = np.linalg.norm(point1 - point4)
        dist2 = np.linalg.norm(point2 - point4)
        dist3 = np.linalg.norm(point3 - point4)
        if(dist1>2*radius and dist2>2*radius and dist3>2*radius):
            break
        R3 = (random.randint(halfStarting+20,halfEnding-20),random.randint(halfStarting1+20,halfEnding1-20))
    image = cv2.circle(image, B3, radius, (255, 0, 0), thickness)
    image = cv2.circle(image, R3, radius, (0,0,255), thickness)

    B4 = (285, 395)
    image = cv2.circle(image, B4, radius, (255, 0, 0), thickness)
    return (B1,B2,B3,B4,R1,R2,R3,image)

def getNormalized(Matrix):

    maxDistance = -1
    for i in Matrix:
        if(maxDistance<i[0]):
            maxDistance = i[0]
    
    for i in Matrix:
        i[0]= i[0]/maxDistance

    return Matrix

def getMaxNormalizedDistance(normalizedDistance):
    choice = normalizedDistance[0][1]
    maxDistance = normalizedDistance[0][0]
    for i in range(1,len(normalizedDistance)):
        if(maxDistance<normalizedDistance[i][0]):
            maxDistance = normalizedDistance[i][0]
            choice = normalizedDistance[i][1]

    return (choice,maxDistance)

def combine(Distance,normalizedDistance,nearRedPlayerDistance,normalizedOppDistance,GoalDistance,normalizedGoalDistance):
    combinedDistance =[]
    for i in range(0,len(normalizedDistance)):
        combinedDistance.append((-1*(normalizedDistance[i][0]*Distance + normalizedOppDistance[i][0]*nearRedPlayerDistance + normalizedGoalDistance[i][0]*GoalDistance),normalizedDistance[i][1]))
    return combinedDistance

def get2ndMinNormalizedDistance(normalizedDistance):

    
    if(normalizedDistance[0][0]<normalizedDistance[1][0]):
        choice = normalizedDistance[0][1]
        minDistance = normalizedDistance[0][0]
        choice2nd = normalizedDistance[1][1]
        minDistance2nd = normalizedDistance[1][0]
    else:
        choice = normalizedDistance[1][1]
        minDistance = normalizedDistance[1][0]
        choice2nd = normalizedDistance[0][1]
        minDistance2nd = normalizedDistance[0][0]
        
    for i in range(2,len(normalizedDistance)):
        if(minDistance>normalizedDistance[i][0]):
            minDistance2nd=minDistance
            choice=choice2nd
            minDistance = normalizedDistance[i][0]
            choice = normalizedDistance[i][1]
        elif(minDistance2nd>normalizedDistance[i][0]):
            minDistance2nd = normalizedDistance[i][0]
            choice2nd = normalizedDistance[i][1]

    return (choice2nd,minDistance2nd)

def bestPathComputation(combineNormalizedDistance,comparePoints,Goal):
    
    utilityForBestPaths = []
    for k in combineNormalizedDistance:
        Utility = k[0]
        n = k[1]
        cmpPoints = comparePoints = [[B1,True],[B2,True],[B3,True],[B4,False],[Goal,False]]
        print("---")
        print(n,Utility)
        for i in cmpPoints:
            if(i[0]==Goal):
                i[1]=True
        for i in cmpPoints:
            if(i[0]==n):
                i[1]=False
        
        while(n != Goal):
            curr = n
            point = np.array(curr)
            matrixDistance = []
            closestRedDistance = []
            goalDistances =[]
            for i in cmpPoints:
                if(i[1]):
                    pointTemp = np.array(i[0])
                    dist = np.linalg.norm(point-pointTemp)
                    matrixDistance.append([dist,i[0]])

                    d = np.linalg.norm(pointTemp-np.array(Goal))
                    goalDistances.append([d,i[0]])

                    minDistance = 999999999
                    for j in oppositePlayers:
                        pointRed = np.array(j)
                        redDistance = np.linalg.norm(pointTemp-pointRed)
                        if(redDistance<minDistance):
                            minDistance=redDistance
                    closestRedDistance.append([1/minDistance,i[0]])

            normalizedDistanceBestPath = getNormalized(matrixDistance)
            normalizedOppDistanceBestPath = getNormalized(closestRedDistance)
            normalizedGoalDistanceBestPath = getNormalized(goalDistances)
            combineNormalizedDistanceBestPath = combine(Distance,normalizedDistanceBestPath,nearRedPlayerDistance,normalizedOppDistanceBestPath,GoalDistance,normalizedGoalDistanceBestPath)

            n,maxUtilityBestPath = getMaxNormalizedDistance(combineNormalizedDistanceBestPath)
            Utility = Utility + maxUtilityBestPath
            print(n,Utility,maxUtilityBestPath)
            for i in cmpPoints:
                if(i[0]==n):
                    i[1]=False
        utilityForBestPaths.append((Utility,k[1]))
    return utilityForBestPaths

imported_Img = Image.open("C:/Users/Gaurav Arya/Desktop/Assignment 1/AI2_Assignment1_T3_2021.PNG")
pixels = imported_Img.load()
s = imported_Img.size
print(s)
# for i in range(100, 300): 
#     pixels[i, 50] = (255, 0, 0)
# imported_Img.show()
# imported_Img.save("C:/Users/Gaurav Arya/Desktop/AI2/pixel_grid.png")

verLine = 2
lastEnding = 0
i=0
starting = 0
ending = 0
halfStarting = 0
halfEnding = 0
while(i<s[0]):
    if(pixels[i,50]==(255,255,255,255)):
        # print("i:",i)
        verLine = verLine -1
        
        j= i
        # print(j)
        while(pixels[j,50]==(255,255,255,255)):
            j=j+1
        lastEnding =j-1
        if(verLine == 1):
            halfStarting = lastEnding + 1
        elif(verLine == 0):
            starting = lastEnding + 1
        elif(verLine == -1):
            ending = i-1
        elif(verLine == -2):
            halfEnding = i-1


        # print(j-1)
        i = lastEnding
        
    i=i+1
print(starting,ending)
print(halfStarting,halfEnding)

verLine = 1
lastEnding = 0
i=0
starting1 = 0
ending1 = 0
halfStarting1 = 0
halfEnding1 = 0
while(i<s[1]):
    if(pixels[100,i]==(255,255,255,255)):
        # print("i:",i)
        verLine = verLine -1
        
        j= i
        # print(j)
        while(j<s[1] and pixels[100,j]==(255,255,255,255)):
            j=j+1
        lastEnding =j-1
        if(verLine == 0):
            starting1 = lastEnding + 1
        elif(verLine == -1):
            ending1 = i-1
            halfStarting1 = lastEnding + 1
        elif(verLine == -2):
            halfEnding1 = i - 1


        # print(j-1)
        i = lastEnding
        
    i=i+1
print(starting1,ending1)
print(halfStarting1,halfEnding1)







(B1,B2,B3,B4,R1,R2,R3,image_withoutBall) = markingPlayers(starting,starting1,ending,ending1,halfStarting,halfStarting1,halfEnding,halfEnding1)
filename = 'C:/Users/Gaurav Arya/Desktop/AI2/Image_withoutBall.PNG'
cv2.imwrite(filename, image_withoutBall)


Goal = (285,10)
Ball = (285, 363)
image_b4 = cv2.circle(image_withoutBall, Ball , 10, (0,0,0), -1)
cv2.imshow('image',image_b4)
cv2.waitKey(0)
cv2.destroyAllWindows()

utilityForBestMove = 0
utilityFor2ndBestMove = 0

Distance = 0.4
nearRedPlayerDistance = 0.4
GoalDistance = 0.2

Current = B4
point = np.array(Current)
matrixDistance = []
comparePoints = [[B1,True],[B2,True],[B3,True],[B4,False],[Goal,False]]
oppositePlayers = [R1,R2,R3]
closestRedDistance = []
goalDistances = []
for i in comparePoints:
    if(i[1]):
        pointTemp = np.array(i[0])
        dist = np.linalg.norm(point-pointTemp)
        matrixDistance.append([dist,i[0]])

        d = np.linalg.norm(pointTemp-np.array(Goal))
        goalDistances.append([d,i[0]])

        minDistance = 999999999
        for j in oppositePlayers:
            pointRed = np.array(j)
            redDistance = np.linalg.norm(pointTemp-pointRed)
            if(redDistance<minDistance):
                minDistance=redDistance
        closestRedDistance.append([1/minDistance,i[0]])

normalizedDistance = getNormalized(matrixDistance)
normalizedOppDistance = getNormalized(closestRedDistance)
normalizedGoalDistance = getNormalized(goalDistances)

combineNormalizedDistance = combine(Distance,normalizedDistance,nearRedPlayerDistance,normalizedOppDistance,GoalDistance,normalizedGoalDistance)
Next,maxUtility = getMaxNormalizedDistance(combineNormalizedDistance)
utilityForBestMove = utilityForBestMove + maxUtility


utilityForBestPaths = bestPathComputation(combineNormalizedDistance,comparePoints,Goal)
comparePoints = [[B1,True],[B2,True],[B3,True],[B4,False],[Goal,False]]

print("Best Move Strategy")
for i in utilityForBestPaths:
    print(i[0],"--",i[1][0],i[1][1])

print("Best Path")

# Next2,minUtility2 = get2ndMinNormalizedDistance(combineNormalizedDistance)
# utilityFor2ndBestMove = utilityFor2ndBestMove + minUtility2

for i in normalizedDistance:
    print(i[0],"--",i[1][0],i[1][1])
print("opp")
for i in normalizedOppDistance:
    print(i[0],"--",i[1][0],i[1][1])
print("Goal")
for i in normalizedGoalDistance:
    print(i[0],"--",i[1][0],i[1][1])
print("combine")
for i in combineNormalizedDistance:
    print(i[0],"--",i[1][0],i[1][1])
print("Next",Next[0],Next[1])
print("Utility",utilityForBestMove)

# print("Next-2nd",Next2[0],Next2[1])
# print(utilityFor2ndBestMove)
for i in comparePoints:
    if(i[0]==Goal):
        i[1]=True
for i in comparePoints:
    if(i[0]==Next):
        i[1]=False


# Displaying
imageTemp = cv2.line(image_b4, Current, Next, (0,0,0), 2)
cv2.imshow('image',imageTemp)
cv2.waitKey(0)
cv2.destroyAllWindows()

imageTemp = cv2.imread("C:/Users/Gaurav Arya/Desktop/AI2/Image_withoutBall.PNG")
imageTemp = cv2.line(imageTemp, Current, Next, (0,0,0), 2)
imageTemp = cv2.circle(imageTemp, (Next[0],Next[1]-30) , 10, (0,0,0), -1)
cv2.imshow('image',imageTemp)
cv2.waitKey(0)
cv2.destroyAllWindows()




while(Next != Goal):
    Current = Next
    point = np.array(Current)
    matrixDistance = []
    closestRedDistance = []
    goalDistances = []

    for i in comparePoints:
        if(i[1]):
            pointTemp = np.array(i[0])
            dist = np.linalg.norm(point-pointTemp)
            matrixDistance.append([dist,i[0]])

            d = np.linalg.norm(pointTemp-np.array(Goal))
            goalDistances.append([d,i[0]])

            minDistance = 999999999
            for j in oppositePlayers:
                pointRed = np.array(j)
                redDistance = np.linalg.norm(pointTemp-pointRed)
                if(redDistance<minDistance):
                    minDistance=redDistance
            closestRedDistance.append([1/minDistance,i[0]])

    normalizedDistance = getNormalized(matrixDistance)
    normalizedOppDistance = getNormalized(closestRedDistance)
    normalizedGoalDistance = getNormalized(goalDistances)

    combineNormalizedDistance = combine(Distance,normalizedDistance,nearRedPlayerDistance,normalizedOppDistance,GoalDistance,normalizedGoalDistance)
    Next,maxUtility = getMaxNormalizedDistance(combineNormalizedDistance)
    utilityForBestMove = utilityForBestMove + maxUtility

    for i in normalizedDistance:
        print(i[0],"--",i[1][0],i[1][1])
    print("opp")
    for i in normalizedOppDistance:
        print(i[0],"--",i[1][0],i[1][1])
    print("Goal")
    for i in normalizedGoalDistance:
        print(i[0],"--",i[1][0],i[1][1])
    print("combine")
    for i in combineNormalizedDistance:
        print(i[0],"--",i[1][0],i[1][1])
    
    print("Next",Next[0],Next[1])
    print("Utility",utilityForBestMove)

    for i in comparePoints:
        if(i[0]==Next):
            i[1]=False
    # Displaying
    imageTemp = cv2.line(imageTemp, Current, Next, (0,0,0), 2)
    cv2.imshow('image',imageTemp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if(Next==Goal):
        constant = 15
    else:
        constant = -30
    imageTemp = cv2.imread("C:/Users/Gaurav Arya/Desktop/AI2/Image_withoutBall.PNG")
    imageTemp = cv2.line(imageTemp, Current, Next, (0,0,0), 2)
    imageTemp = cv2.circle(imageTemp, (Next[0],Next[1]+constant) , 10, (0,0,0), -1)
    cv2.imshow('image',imageTemp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(utilityForBestMove)
# image_Tb3 = cv2.line(image_b4, B4, B3, (0,0,0), 2)
# cv2.imshow('image',image_Tb3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # (B1,B2,B3,B4,image_withoutBall) = markingPlayers(starting,starting1,ending,ending1,halfStarting,halfStarting1,halfEnding,halfEnding1)
# image_b3 = cv2.imread("C:/Users/Gaurav Arya/Desktop/AI2/Image_withoutBall.PNG")
# image_b3 = cv2.line(image_b3, B4, B3, (0,0,0), 2)
# image_b3 = cv2.circle(image_b3, (B3[0],B3[1]-30) , 10, (0,0,0), -1)
# cv2.imshow('image',image_b3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image_Tb2 = cv2.line(image_b3, B3, B2, (0,0,0), 2)
# cv2.imshow('image',image_Tb2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image_b2 = cv2.imread("C:/Users/Gaurav Arya/Desktop/AI2/Image_withoutBall.PNG")
# image_b2 = cv2.line(image_b2, B3, B2, (0,0,0), 2)
# image_b2 = cv2.circle(image_b2, (B2[0],B2[1]-30) , 10, (0,0,0), -1)
# cv2.imshow('image',image_b2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image_Tb1 = cv2.line(image_b2, B2, B1, (0,0,0), 2)
# cv2.imshow('image',image_Tb1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image_b1 = cv2.imread("C:/Users/Gaurav Arya/Desktop/AI2/Image_withoutBall.PNG")
# image_b1 = cv2.line(image_b1, B2, B1, (0,0,0), 2)
# image_b1 = cv2.circle(image_b1, (B1[0],B1[1]-30) , 10, (0,0,0), -1)
# cv2.imshow('image',image_b1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# image_Tb0 = cv2.line(image_b1, B1, Goal, (0,0,0), 2)
# cv2.imshow('image',image_Tb0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image_b0 = cv2.imread("C:/Users/Gaurav Arya/Desktop/AI2/Image_withoutBall.PNG")
# image_b0 = cv2.line(image_b0, B1, Goal, (0,0,0), 2)
# image_b0 = cv2.circle(image_b0, (Goal[0],Goal[1]+15) , 10, (0,0,0), -1)
# cv2.imshow('image',image_b0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()