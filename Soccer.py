from PIL import Image
from IPython.display import display
import cv2
import random
import numpy as np


def sensingTheField():

    imported_Img = Image.open("C:/Users/Gaurav Arya/Desktop/Assignment 1/AI2_Assignment1_T3_2021.PNG")
    pixels = imported_Img.load()
    s = imported_Img.size

    verLine = 2
    lastEnding = 0
    i=0
    starting = 0
    ending = 0
    halfStarting = 0
    halfEnding = 0
    while(i<s[0]):
        if(pixels[i,50]==(255,255,255,255)):
            
            verLine = verLine -1
            
            j= i
            
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


            
            i = lastEnding
            
        i=i+1


    verLine = 1
    lastEnding = 0
    i=0
    starting1 = 0
    ending1 = 0
    halfStarting1 = 0
    halfEnding1 = 0
    while(i<s[1]):
        if(pixels[100,i]==(255,255,255,255)):
            
            verLine = verLine -1
            
            j= i
            
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


            
            i = lastEnding
            
        i=i+1

    return (starting,starting1,ending,ending1,halfStarting,halfStarting1,halfEnding,halfEnding1)

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
    image = cv2.putText(image, "B1", (B1[0]-14,B1[1]+10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

    image = cv2.circle(image,R1 , radius, (0,0,255), thickness)
    image = cv2.putText(image, "R1", (R1[0]-14,R1[1]+10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)


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
    image = cv2.putText(image, "B2", (B2[0]-14,B2[1]+10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    image = cv2.circle(image, R2, radius, (0,0,255), thickness)
    image = cv2.putText(image, "R2", (R2[0]-14,R2[1]+10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

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
    image = cv2.putText(image, "B3", (B3[0]-14,B3[1]+10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    image = cv2.circle(image, R3, radius, (0,0,255), thickness)
    image = cv2.putText(image, "R3", (R3[0]-14,R3[1]+10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)

    B4 = (285, 395)
    image = cv2.circle(image, B4, radius, (255, 0, 0), thickness)
    image = cv2.putText(image, "B4", (B4[0]-14,B4[1]+10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    return (B1,B2,B3,B4,R1,R2,R3,image)

def AllPathComputation(weights,allPossiblePoints,Goal,B4):
    
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

            d = np.linalg.norm(pointTemp-np.array(Goal)) + 1
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

    combineNormalizedDistance = combine(weights[0],normalizedDistance,weights[1],normalizedOppDistance,weights[2],normalizedGoalDistance)

    Visted = []
    Visted.append(B4)
    for k in combineNormalizedDistance:
        Utility = k[0]
        n = k[1]
        Visted.append(n)
        recursiveAllPathComputation(Visted,Utility,Goal,allPossiblePoints,oppositePlayers)
        Visted.pop(len(Visted)-1)

    return 

def recursiveAllPathComputation(Visited,Utility,Goal,allPossiblePoints,oppositePlayers):
    if(Visited[len(Visited)-1]==Goal):
        xx = []
        for i in Visited:
            xx.append(i)
        xx.append(Utility)
        allPath.append(xx)
        return
    else:
        
        curr = Visited[len(Visited)-1]
        point = np.array(curr)
        matrixDistance = []
        closestRedDistance = []
        goalDistances =[]

        for i in allPossiblePoints:
            if(i not in Visited):
                pointTemp = np.array(i)
                dist = np.linalg.norm(point-pointTemp)
                matrixDistance.append([dist,i])

                d = np.linalg.norm(pointTemp-np.array(Goal)) + 1
                goalDistances.append([d,i])
                minDistance = 999999999
                for j in oppositePlayers:
                    pointRed = np.array(j)
                    redDistance = np.linalg.norm(pointTemp-pointRed)
                    if(redDistance<minDistance):
                        minDistance=redDistance
                closestRedDistance.append([1/minDistance,i])
        
        normalizedDistanceBestPath = getNormalized(matrixDistance)
        normalizedOppDistanceBestPath = getNormalized(closestRedDistance)
        normalizedGoalDistanceBestPath = getNormalized(goalDistances)
        combineNormalizedDistanceBestPath = combine(Distance,normalizedDistanceBestPath,nearRedPlayerDistance,normalizedOppDistanceBestPath,GoalDistance,normalizedGoalDistanceBestPath)

        for i in combineNormalizedDistanceBestPath:
            Visited.append(i[1])
            Utility = Utility + i[0]
            recursiveAllPathComputation(Visited,Utility,Goal,allPossiblePoints,oppositePlayers)
            Visited.pop(len(Visited)-1)
            Utility = Utility - i[0]
        
        return

def getNormalized(Matrix):

    maxDistance = -1
    for i in Matrix:
        if(maxDistance<i[0]):
            maxDistance = i[0]
    
    for i in Matrix:
        i[0]= i[0]/maxDistance

    return Matrix

def combine(Distance,normalizedDistance,nearRedPlayerDistance,normalizedOppDistance,GoalDistance,normalizedGoalDistance):
    combinedDistance =[]
    for i in range(0,len(normalizedDistance)):
        combinedDistance.append((-1*(normalizedDistance[i][0]*Distance + normalizedOppDistance[i][0]*nearRedPlayerDistance + normalizedGoalDistance[i][0]*GoalDistance),normalizedDistance[i][1]))
    return combinedDistance

def identifyingTop2Paths(Allpaths):
    bestPathindex = 0
    Utility_bestPathindex = -9999999
    best2ndPathindex = 0
    Utility_best2ndPathindex = -999999

    for i in range(0,len(Allpaths)):
        u = Allpaths[i][len(Allpaths[i])-1]
        if(u>Utility_bestPathindex):
            Utility_best2ndPathindex = Utility_bestPathindex
            best2ndPathindex =  bestPathindex
            Utility_bestPathindex =  u
            bestPathindex = i
        elif(u>Utility_best2ndPathindex):
            Utility_best2ndPathindex = u
            best2ndPathindex =  i
    
    return (bestPathindex,Utility_bestPathindex,best2ndPathindex,Utility_best2ndPathindex)

(starting,starting1,ending,ending1,halfStarting,halfStarting1,halfEnding,halfEnding1) = sensingTheField()
(B1,B2,B3,B4,R1,R2,R3,image_withoutBall) = markingPlayers(starting,starting1,ending,ending1,halfStarting,halfStarting1,halfEnding,halfEnding1)
filename = 'C:/Users/Gaurav Arya/Desktop/AI2/Image_withoutBall.PNG'
cv2.imwrite(filename, image_withoutBall)

Goal = (285,10)
Ball = (285, 363)


Distance = 0.5
nearRedPlayerDistance = 0.4
GoalDistance = 0.1

pointDict = {B1:"B1",B2:"B2",B3:"B3",B4:"B4",Goal:"Goal"}


allPath =[]
AllPathComputation((Distance,nearRedPlayerDistance,GoalDistance),(B1,B2,B3,B4,Goal),Goal,B4)

# print("All Paths")
# for i in range(0,len(allPath)):
#     print("Path"+str(i))
#     print(allPath[i])
    
(bestPathindex,Utility_bestPathindex,best2ndPathindex,Utility_best2ndPathindex) = identifyingTop2Paths(allPath)
print("\n2nd - Best Path")
for i in range(0,len(allPath[best2ndPathindex])-1):
    if(i==len(allPath[best2ndPathindex])-2):
        print(pointDict[allPath[best2ndPathindex][i]],end ="    ")
    else:
        print(pointDict[allPath[best2ndPathindex][i]],"-->",end =" ")
print("Utility = ",Utility_best2ndPathindex)

print("Best Path")
for i in range(0,len(allPath[bestPathindex])-1):
    if(i==len(allPath[bestPathindex])-2):
        print(pointDict[allPath[bestPathindex][i]],end ="    ")
    else:
        print(pointDict[allPath[bestPathindex][i]],"-->",end =" ")
print("Utility = ",Utility_bestPathindex)



Passes =[]
font = cv2.FONT_HERSHEY_TRIPLEX
org = (230, 500)
fontScale = 1
color = (0,140,255)
thickness = 2

# Displaying

image_b4 = cv2.circle(image_withoutBall, Ball , 10, (0,0,0), -1)
cv2.imshow('image',image_b4)
cv2.waitKey(0)
cv2.destroyAllWindows()


passNumber = 1
Current = allPath[bestPathindex][0]
Next = allPath[bestPathindex][1]
Passes.append((Current,Next))
text = "Pass :" + str(passNumber)
imageTemp = cv2.line(image_b4, Current, Next, (0,0,0), 1)
imageTemp = cv2.putText(imageTemp, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow('image',imageTemp)
cv2.waitKey(0)
cv2.destroyAllWindows()

imageTemp = cv2.imread("C:/Users/Gaurav Arya/Desktop/AI2/Image_withoutBall.PNG")
imageTemp = cv2.line(imageTemp, Current, Next, (0,0,0), 1)
imageTemp = cv2.circle(imageTemp, (Next[0],Next[1]-30) , 10, (0,0,0), -1)
cv2.imshow('image',imageTemp)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(1,len(allPath[bestPathindex])-2):
    Current = allPath[bestPathindex][i]
    Next = allPath[bestPathindex][i+1]

    if(Next != Goal):
        passNumber = passNumber +1 
        text = "Pass :" + str(passNumber)
    else:
        text = "Goal !!"


    Passes.append((Current,Next))
    for i in Passes:
        imageTemp = cv2.line(imageTemp, i[0], i[1], (0,0,0), 1)
    imageTemp = cv2.putText(imageTemp, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('image',imageTemp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if(Next==Goal):
        constant = 15
    else:
        constant = -30
    imageTemp = cv2.imread("C:/Users/Gaurav Arya/Desktop/AI2/Image_withoutBall.PNG")
    for i in Passes:
        imageTemp = cv2.line(imageTemp, i[0], i[1], (0,0,0), 1)
    imageTemp = cv2.circle(imageTemp, (Next[0],Next[1]+constant) , 10, (0,0,0), -1)
    cv2.imshow('image',imageTemp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


