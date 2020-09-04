# Facial-Recognition-SVM-Facenet

This is the final version of the three notebooks. First, FR_Detection, uses MTCNN and reads the faces out of each picture, extracts the pixel values as numpy arrays and saves these in a compressed .npz file(four variables, trainX, trainy, testX and testy).
Then the FR_Embedding notebook is run, wherein a trained Facenet model is used to generate embeddings(128-dimensional) for each face, previously extracted. These are then stored in another .npz file, to be directly loaded and trained a classifier(so as to make tweaking of the final model separate from the preprocessing). 
The final notebook, FR_Classification, trains an Support Vector Machine(SVM) Classifier. It uses a polynomial kernel of degree 2 and the values of C and coef1 have been set in order to reduce variance and overfitting as far as possible, so that the model is able to generalise well. 
