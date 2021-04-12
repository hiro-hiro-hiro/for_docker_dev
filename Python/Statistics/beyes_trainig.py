import numpy as np

class NaiveBayes1(object):
    """
    Naive Bayes class (1)
    """
    def __init__(self):
        """
        Constructor
        """
        # 配列の大きさは不明なのでNoneで初期化 
        self.pY_ = None # Pr[y]
        self.pXgY_ = None # Pr[xj|y]
    

    def fit(self, X, y):
        """
        Fitting model
        """
        # 2.4.1 定数の設定
        n_samples = X.shape[0]
        n_features = X.shape[1]
        # 0か1の二値変数
        n_classes = 2
        n_fvalues = 2

        if n_samples != len(y):
            raise ValueError('Mismatched number of samples.')
        
        # 2.4.2 クラスの分布の学習
        # Pr[y] = N[yi=y] / N
        # 各クラスごとに発生する事例の数え上げ N[yi=y]
        nY = np.zeros(n_classes, dtype=int)
        for i in range(n_samples):
            nY[y[i]] += 1
        # 各クラスの確率を格納 Pr[y]
        self.pY_ = np.empty(n_classes, dtype=float)
        for i in range(n_classes):
            self.pY_[i] = nY[i] / n_samples

        # 2.4.3 特徴の分布の学習
        # Pr[xj|y] = N[xij=xj,yi=y] / N[yi=y]
        nXY = np.zeros((n_features, n_fvalues, n_classes), dtype=int)
        for i in range(n_samples):
            for j in range(n_features):
                nXY[j, X[i, j], y[i]] += 1
        
        self.pXgY_ = np.empty((n_features, n_samples, n_classes), dtype=int)
        for j in range(n_features):
            for xi in range(n_fvalues):
                for yi in range(n_classes):
                    self.pXgY_[j, xi, yi] = nXY[j, xi, yi] / float(nY[yi])
        pass


    def predict(self, X):
        """
        Predict class
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        y = np.empty(n_samples, dtype=int)

        """
        # ループ処理による対数同時確率の計算
        logpXY = np.log(self.pY_) # 第一項
        for j in range(n_features):
            logpXY = logpXY + np.log(self.pXgY_[j, xi[j], :]) # 第二項
        """
        # y^ = argmax_y(logPr[y] + Σ_jlogPr[xnewj[y]])
        # numpyの配列処理による計算
        for i, xi in enumerate(X):
            logpXY = (np.log(self.pY_) +
                      np.sum(np.log(self.pXgY_[np.arange(n_features), xi, :])), axis=0)
            y[i] = np.argmax(logpXY)
        pass




