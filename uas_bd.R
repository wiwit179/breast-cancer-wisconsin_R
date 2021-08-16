dataset = read.csv("D:\\Big Data\\uas\\breast-cancer-wisconsin.csv", header = TRUE)

dim(dataset)

names(dataset)

str(dataset)

summary(dataset)

barplot(dataset$Class)

library(Hmisc)
describe(dataset$Class)

sum(is.na(dataset[]))

#fitur selection
library(Boruta)
library(randomForest)
library(mlbench)
library(caret)
set.seed(10)
boruta <- Boruta(Class ~ ., dataset, doTrace = 2, maxRuns = 20)
print(boruta)
plot(boruta)

bor <- TentativeRoughFix(boruta)
print(bor)
attStats(boruta)

#reduksi dataset
data <- dataset[,2:11]
print(data)

#pembagian data dengan 2 FOld
#part1
data.test.1 = data[1:70,]
data.train.1 = data[71:350,]
#part2
data.test.2 = data[351:420,]
data.train.2 = data[421:699,]

#data test
data.test = rbind(data.test.1, data.test.2)
str(data.test)
#data training
data.train <- rbind(data.train.1, data.train.2)
data.train$Class <- as.factor(data.train$Class)
str(data.train)
data.test$Class <- as.factor(data.test$Class)
str(data.train)

#pembagian data dengan 7 FOld
#part1
data.test.1 = data[1:20,]
data.train.1 = data[21:100,]
#part2
data.test.2 = data[101:120,]
data.train.2 = data[121:200,]
#part3
data.test.3 = data[201:220,]
data.train.3 = data[221:300,]
#part4
data.test.4 = data[301:320,]
data.train.4 = data[321:400,]
#part5
data.test.5 = data[401:420,]
data.train.5 = data[421:500,]
#part6
data.test.6 = data[501:520,]
data.train.6 = data[521:600,]
#part7
data.test.7 = data[601:620,]
data.train.7 = data[621:699,]

#data test
data.test = rbind(data.test.1, data.test.2, data.test.3, data.test.4, 
                  data.test.5, data.test.6, data.test.7)
str(data.test)
#data training
data.train <- rbind(data.train.1, data.train.2, data.train.3, data.train.4, 
                  data.train.5, data.train.6, data.train.7)
data.train$Class <- as.factor(data.train$Class)
str(data.train)
data.test$Class <- as.factor(data.test$Class)
str(data.train)

#pembagian data dengan 4 FOld
#part1
data.test.1 = data[1:35,]
data.train.1 = data[36:175,]
#part2
data.test.2 = data[176:210,]
data.train.2 = data[211:350,]
#part3
data.test.3 = data[351:385,]
data.train.3 = data[386:525,]
#part4
data.test.4 = data[526:560,]
data.train.4 = data[561:699,]

#data test
data.test = rbind(data.test.1, data.test.2, data.test.3, data.test.4)
str(data.test)
#data training
data.train <- rbind(data.train.1, data.train.2, data.train.3, data.train.4)
data.train$Class <- as.factor(data.train$Class)
str(data.train)
data.test$Class <- as.factor(data.test$Class)
str(data.train)

#pembagian data dengan 10 FOld
#part1
data.test.1 = data[1:15,]
data.train.1 = data[16:70,]
#part2
data.test.2 = data[71:85,]
data.train.2 = data[86:140,]
#part3
data.test.3 = data[141:155,]
data.train.3 = data[156:210,]
#part4
data.test.4 = data[211:225,]
data.train.4 = data[256:280,]
#part5
data.test.5 = data[281:295,]
data.train.5 = data[296:350,]
#part6
data.test.6 = data[351:365,]
data.train.6 = data[366:420,]
#part7
data.test.7 = data[421:435,]
data.train.7 = data[436:490,]
#part8
data.test.8 = data[491:505,]
data.train.8 = data[506:560,]
#part9
data.test.9 = data[561:575,]
data.train.9 = data[576:630,]
#part10
data.test.10 = data[631:645,]
data.train.10 = data[646:699,]
#data test
data.test = rbind(data.test.1, data.test.2, data.test.3, data.test.4, 
                  data.test.5, data.test.6, data.test.7, data.test.8, 
                  data.test.9, data.test.10)
str(data.test)
#data training
data.train <- rbind(data.train.1, data.train.2, data.train.3, data.train.4, 
                    data.train.5, data.train.6, data.train.7, data.train.8, 
                    data.train.9, data.train.10)
data.train$Class <- as.factor(data.train$Class)
str(data.train)
data.test$Class <- as.factor(data.test$Class)
str(data.train)

library(RWeka)
#membuat model Decision Tree
model = J48(Class~.,data.train)
#membuat prediksi
pred = predict(model, data.test)
#menghitung kinerja
kinerja.values = confusionMatrix(pred, data.test$Class)
print(kinerja.values)

#pohon keputusan
library(party)
library(tree)
library(rpart)
library(rpart.plot)
tree.data = rpart(Class~.,data.train)
rpart.plot(tree.data)
