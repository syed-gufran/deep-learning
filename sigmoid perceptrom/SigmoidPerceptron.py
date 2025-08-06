import numpy as np
class SigmoidPerceptron():

    def __init__(self ,input_size):

        self.weight = np.random.randn(input_size)
        self.bias = np.random.randn(1) 
    
    def sigmoid(self , z):
        return 1/(1+np.exp(-z))

    def predict(self,inputs):
        weighted_sum =np.dot(inputs,self.weight)+self.bias
        return self.sigmoid(weighted_sum)

    def fit(self,inputs , targets ,learning_rate ,num_epochs):

        num_examples= inputs.shape[0]
    
        for epoch in range(num_epochs):
            for i in range(num_examples):
                input_vector = inputs[i]
                target = targets[i]
                prediction = self.predict(input_vector)
                error = target -prediction
    
                gradient_weight = error*prediction *(1 - prediction)*input_vector
                self.weight += learning_rate*gradient_weight
    
                gradient_bias = error*prediction*(1-prediction)
                self.bias += learning_rate * gradient_bias


    def evaluate(self , inputs , targets):
        correct =0 
        for input_vector , target in zip(inputs, targets):
            prediction = self.predict(input_vector)

            if prediction>=0.5:
                predicted_class =1
            else:
                predicted_class = 0

            if predicted_class==target:
                correct+=1

        accuracy =  correct/len(targets)
        return accuracy 