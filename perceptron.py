class Perceptron:
    def __init__(self, lr=0.1, epochs=10):
        self.lr, self.epochs = lr, epochs
        self.w, self.b = [], 0.0

    def predict(self, x):
        total = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return 1 if total >= 0 else 0

    def fit(self, X, y):
        self.w = [0.0] * len(X[0])
        for _ in range(self.epochs):
            for x_i, y_i in zip(X, y):
                error = y_i - self.predict(x_i)
                self.w = [wj + self.lr * error * xi for wj, xi in zip(self.w, x_i)]
                self.b += self.lr * error

# --- 1. Load Data From File ---
X, y = [], []
with open("dataset.csv", "r") as file:
    for line in file:
        line = line.strip()
        if line:  # Skip empty lines
            row = [float(val) for val in line.split(",")]
            X.append(row[:-1])   # All items except the last are features
            y.append(int(row[-1])) # The very last item is the label (0 or 1)

# --- 2. Train Model ---
model = Perceptron(lr=0.1, epochs=10)
model.fit(X, y)

# --- 3. Final Outputs ---
print(f"Weights: {model.w} | Bias: {model.b}\n")
print("--- Final Performance ---")
for x_i, y_i in zip(X, y):
    pred = model.predict(x_i)
    print(f"Input: {x_i} | Target: {y_i} | Predicted: {pred} | {'no change' if pred == y_i else 'change'}")