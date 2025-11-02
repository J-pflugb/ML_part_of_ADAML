from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec
import math
import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.io import arff
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
print("in")



#.frame.info()
def task_1(include_all=True): # new data set
    # Task 1
    #bikes=fetch_openml(data_id=42712)
    #bikes.frame.info()
    #print(bikes.frame.head())
    #print(bikes.frame.columns)
    #frame_bikes=bikes.frame
    data, meta = arff.loadarff('dataset.arff')
    print(meta)

    frame_bikes = pd.DataFrame(data)
    print(frame_bikes.columns)
    print(frame_bikes.info())
    if include_all:
        le = LabelEncoder()
        for col in frame_bikes.select_dtypes(include='object').columns:
            frame_bikes[col] = le.fit_transform(frame_bikes[col])
    data_numeric=frame_bikes.select_dtypes(include=['number'])
    print(data_numeric.columns)
    print(data_numeric.columns)
    X = data_numeric.copy().drop(columns=['count',"casual","registered"])#", season","holiday","workingday","weather",'year',"month","hour","weekday"])#  # all numeric feature columns
    Y = data_numeric['count']
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    print(X_sc.shape)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sc)
    tsne = TSNE(n_components=2, perplexity=100, random_state=42)
    X_tsne = tsne.fit_transform(X_sc)
    print(X_tsne)

    loadings = pd.DataFrame(pca.components_.T,
                            columns=['PC1', 'PC2'],
                            index=X.columns)
    print(loadings)
    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    scatter = ax[0].scatter(X_pca[:,0], X_pca[:,1], c=Y, cmap='viridis')
    ax[0].set_title('PCA')
    fig.colorbar(scatter, ax=ax[0], label='Bike Count')

    # t-SNE scatter colored by count
    scatter = ax[1].scatter(X_tsne[:,0], X_tsne[:,1], c=Y, cmap='viridis')
    ax[1].set_title('t-SNE')
    fig.colorbar(scatter, ax=ax[1], label='Bike Count')

    plt.show()

    fig, ax =  plt.subplots(4, int(np.ceil(X_sc.shape[1]/2)),figsize=(16,9))#2


    for i in range(X_sc.shape[1]):
        """k=int(np.floor(i/4))
        j=int(i%4) 
        print(k,j)"""
        if i<np.ceil(X_sc.shape[1]/2):
            k=0
        else:
            k=2
        scatter = ax[k+0, i%int(np.ceil(X_sc.shape[1]/2))].scatter(X_pca[:, 0], X_pca[:, 1], c=X_sc[:, i], cmap='viridis')
        ax[k+0, i%int(np.ceil(X_sc.shape[1]/2))].set_title(X.columns[i])
        #fig.colorbar(scatter, ax=ax[0, i], label=X.columns[i])
        scatter = ax[k+1,i%int(np.ceil(X_sc.shape[1]/2))].scatter(X_tsne[:,0], X_tsne[:,1], c=X_sc[:,i], cmap='viridis')
        #ax[k+1,i].set_title(X.columns[i])
        #fig.colorbar(scatter, ax=ax[1,i], label=X.columns[i])
    plt.tight_layout()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X_sc, Y, test_size=0.2, random_state=42)
    X_pca_train, X_pca_test, _, _ = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
    X_tsne_train, X_tsne_test, _, _ = train_test_split(X_tsne, Y, test_size=0.2, random_state=42)

    # Initialize model
    rf = RandomForestRegressor(n_estimators=50,random_state=10)

    # Function to train & evaluate
    def train_evaluate(X_tr, X_te, y_tr, y_te, name):
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        mse = mean_squared_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)
        print(f"{name} -> MSE: {mse:.2f}, R2: {r2:.2f}")

    # Original features
    train_evaluate(X_train, X_test, y_train, y_test, "Original Features")

    # PCA
    train_evaluate(X_pca_train, X_pca_test, y_train, y_test, "PCA (2 components)")

    # t-SNE
    train_evaluate(X_tsne_train, X_tsne_test, y_train, y_test, "t-SNE (2 components)")


def task_2():
    "https://github.com/JustGlowing/minisom/blob/master/examples/BasicUsage.ipynb"
    #numbers = fetch_openml(data_id=554)
    #numbers.frame.info()
    data, meta = arff.loadarff('mnist_784.arff')
    df = pd.DataFrame(data)
    print(df.head())
    print(df.shape)
    print(df.iloc[:,0].values)
    df_=df
    #df_=df.iloc[:1000,:] for initial visualizing testing
    X = df_.values[:,:-1]
    Y = df_["class"].astype(int).values
    print(Y)
    label_names = {0:"0",1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    print(X.shape)
    print(data[0])
    print(data.shape)
    data = scale(X)
    print(data.shape)

    #Y=Y.values[:1000]
    data = np.real(data).astype(np.float32)
    n_neurons = 15
    m_neurons = 15
    som = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5, random_seed=10,)
    som.random_weights_init(data)
    som.train(data, 50000,random_order=True, verbose=True)  # random training
    keys_initial = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    matrix = np.empty((n_neurons, m_neurons), dtype=object)
    for i in range(n_neurons):
        for j in range(m_neurons):
            matrix[i, j] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
            #had problems with numpy.array that's why the dict
    for x, t in zip(data, Y):  # scatterplot
        w = som.winner(x)
        #print(w[0], w[1])
        #print(matrix[w[0], w[1]])
        matrix[w[0], w[1]][t] += 1
    print(matrix)
    matrix_winner = np.zeros((n_neurons, m_neurons), dtype=float)
    matrix_values = np.zeros((n_neurons, m_neurons), dtype=float)
    matrix_purity = np.zeros((n_neurons, m_neurons), dtype=float)
    for i in range(n_neurons):
        for j in range(m_neurons):
            print(sum(matrix[i, j].values()),matrix[i, j].values())
            max_value= max(matrix[i, j], key=matrix[i, j].get)
            total = sum(matrix[i, j].values())
            if total == 0:
                matrix_winner[i, j]=np.nan
                matrix_values[i, j] = np.nan
                matrix_purity[i, j] = np.nan
            else:
                matrix_winner[i, j] = max_value
                matrix_values[i, j] = total
                matrix_purity[i, j] = (matrix[i, j][max_value] / total) * 100

    plt.imshow(matrix_winner, cmap='viridis',origin='lower', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(f'Winning number')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    #plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    for i in range(n_neurons):
        for j in range(m_neurons):

                plt.text(j, i, f"{matrix_winner[i, j]:.0f}", ha='center', va='center', color='white')

    plt.show()
    plt.imshow(matrix_values, cmap='viridis',origin='lower', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title("Number of Values")
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    #plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    for i in range(n_neurons):
        for j in range(m_neurons):
            plt.text(j, i, f"{matrix_values[i, j]:.0f}", ha='center', va='center', color='white')
    plt.show()
    plt.imshow(matrix_purity, cmap='viridis',origin='lower', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title("Purity of Cell")
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    #plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    for i in range(n_neurons):
        for j in range(m_neurons):
            plt.text(j, i, f"{matrix_purity[i, j]:.2f}", ha='center', va='center', color='white')
    plt.show()

    labels_map = som.labels_map(data, [label_names[t] for t in Y])

    fig = plt.figure(figsize=(9, 9))
    the_grid = gridspec.GridSpec(n_neurons, m_neurons, fig)
    for position in labels_map.keys():
        label_fracs = [labels_map[position][l] for l in label_names.values()]
        plt.subplot(the_grid[position[0],
        position[1]], aspect=1)
        patches, texts = plt.pie(label_fracs)

    plt.legend(patches, label_names.values(), bbox_to_anchor=(3.5, 6.5), ncol=3)
    plt.show()



    plt.figure(figsize=(10, 10))

    for i in range(n_neurons):
        for j in range(m_neurons):
            # Get the weights (prototype) for this SOM node
            plt.subplot(n_neurons, m_neurons, i * m_neurons + j + 1)
            plt.axis('off')
            plt.imshow(som.get_weights()[i, j].reshape(28, 28), cmap='gray')

    plt.tight_layout()
    plt.show()


if "__main__" == __name__:
    #task_2()
    task_1()
