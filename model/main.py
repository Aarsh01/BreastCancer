import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.pipeline import make_pipeline
import pickle

def create_model(data): 
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # Create a pipeline for scaling and PCA
    pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=9, random_state=42))

    # Fit and transform X with PCA
    X_pca = pca_pipeline.fit_transform(X)

    # Resampling the minority class using SMOTETomek
    smt_final = SMOTETomek(sampling_strategy='minority', random_state=42)
    X_res_final, y_res_final = smt_final.fit_resample(X_pca, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_res_final, y_res_final, test_size=0.2, random_state=42)

    # Train the final model (Random Forest Classifier)
    final_model = RandomForestClassifier()
    final_model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = final_model.predict(X_test)
    print("Final Random Forest Classifier Accuracy Score (Train):", final_model.score(X_train, y_train))
    print("Final Random Forest Classifier Accuracy Score (Test):", round(accuracy_score(y_pred, y_test)*100,3))
        
    return final_model, pca_pipeline

def get_clean_data():
    data = pd.read_csv("data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def main():
    data = get_clean_data()
    model, pipeline = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('model/pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)


if __name__ == '__main__':
    main()
